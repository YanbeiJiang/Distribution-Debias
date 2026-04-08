import torch
from preference_datasets import get_batch_iterator
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import scipy
import torch.nn.functional as F
import numpy as np
import json
import torch._dynamo
from peft import PeftModel
torch._dynamo.config.suppress_errors = True

def inference(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    #print(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, torch_dtype="bfloat16", cache_dir="/data/projects/punim1996/anaconda3/.cache/hub")
    model = PeftModel.from_pretrained(model, args.ckpt)
    #model = PeftModel.from_pretrained(model, args.ckpt)
    # add padding token
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # load state from policy.pt
    # state = torch.load(args.ckpt, map_location='cuda:0')
    # model.load_state_dict(state['state'])
    model.cuda()
    model.eval()

    with open(args.datasets, "r") as f:
        test_data = json.load(f)
    
    # Generate prompts
    def build_prompt(messages):
        return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages if msg['content'])

    # Inference
    results = []

    for example in tqdm(test_data):
        prompt = example["messages"][0]["content"]
        #print(prompt)
        messages = [
            {"role": "user", "content": f"<Q>{prompt}<A>"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        #qprint(text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 第一步：生成一个token来获取"0"和"1"的概率
        with torch.no_grad():
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[0, -1, :]  # 获取最后一个位置的logits

        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        # ]
        # # max_indices = torch.argmax(next_token_logits, dim=-1)
        # # print(max_indices)
        # response = tokenizer.batch_decode(next_token_logits, skip_special_tokens=True)[0]
        # print(response)

        if "gender" in args.output_dir:
            num_classes = 2
        elif "race" in args.output_dir:
            num_classes = 5
        elif "sentiment" in args.output_dir:
            num_classes = 3
        else:
            raise ValueError(f"Unknown task type in output_dir: {args.output_dir}")

        # # 获取每个类别对应的 token id
        class_tokens = [str(i) for i in range(num_classes)]
        token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in class_tokens]

        # 提取对应 token 的 logits
        logits = torch.stack([next_token_logits[token_id] for token_id in token_ids])
        #print(logits)
        # 计算 softmax 概率
        probs = F.softmax(logits, dim=0)

        # 转成 Python float 列表，便于记录或打印
        prob_list = [p.item() for p in probs]

        # 输出各类别概率（可选）
        # for i, p in enumerate(prob_list):
        #     print(f"Probability of '{i}': {p:.4f}")

        # # 根据概率采样预测类别
        sampled_token_id = torch.multinomial(probs, 1).item()
        sampled_digit = str(sampled_token_id)
        #print("sampled digit: ", sampled_digit)
        # 第二步：将采样的结果添加到prompt中，然后正常生成
        updated_messages = [
            {"role": "user", "content": f"<Q>{prompt}<A>{sampled_digit}"}
        ]
        updated_text = tokenizer.apply_chat_template(
            updated_messages,
            tokenize=False,
            add_generation_prompt=True
        )
       #q print(updated_text)
        updated_model_inputs = tokenizer([updated_text], return_tensors="pt").to(model.device)
        
        # 正常生成剩余的tokens
        if args.top_p != 0.0:
            generated_ids = model.generate(
                **updated_model_inputs,
                max_new_tokens=256,  # 减少1因为我们已经添加了一个digit
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                top_p = args.top_p
            )
        elif args.top_k != 0.0:
            args.top_k = int(args.top_k)
            generated_ids = model.generate(
                **updated_model_inputs,
                max_new_tokens=256,  # 减少1因为我们已经添加了一个digit
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                top_k = args.top_k
            )
        elif args.temperature != 0.0:
            generated_ids = model.generate(
                **updated_model_inputs,
                max_new_tokens=256,  # 减少1因为我们已经添加了一个digit
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature = args.temperature
            )
        else:
            generated_ids = model.generate(
                **updated_model_inputs,
                max_new_tokens=256,  # 减少1因为我们已经添加了一个digit
                return_dict_in_generate=True,
                output_scores=True
            )

        output_ids = generated_ids.sequences
        generated_token_ids = [
            output_ids[i][len(updated_model_inputs.input_ids[i]):] for i in range(len(output_ids))
        ]
        # logits_scores = generated_ids.scores
        
        # # 如果需要完整的生成结果（包括采样的digit）
        response = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0].replace("0","").replace("1","").replace("2","").replace("3","").replace("4","")

        #print(response)
        
        results.append({
            "index": example["index"],
            "messages": example["messages"],
            "response": response
        })

    # Optionally save results
    with open(args.output_dir, "w") as f:
        json.dump(results, f, indent=2)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    argparser.add_argument('--ckpt', type=str, default='/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_real_uk/Qwen2.5-7B')
    argparser.add_argument('--output_dir', type=str, default='/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/uk_test_GDPO-qwen-7b_v2.json')
    argparser.add_argument('--datasets', type=str, default='/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v2.json')
    argparser.add_argument('--top_p', type=float, default=0.0)
    argparser.add_argument('--top_k', type=float, default=0.0)
    argparser.add_argument('--temperature', type=float, default=0.0)
    args = argparser.parse_args()
    #print(args)
    inference(args)

# main function
if __name__ == '__main__':
    main()
        
        
    