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
import pandas as pd

def inference(args):
    if "uk" in args.datasets:
        gt_df = pd.read_csv("uk_shortlist.csv")
    elif "us" in args.datasets:
        gt_df = pd.read_csv("us_shortlist.csv")
    # gt_male proportion = male_count / (male_count + female_count)
    gt_df["gt_male"] = gt_df["male_count"] / (gt_df["male_count"] + gt_df["female_count"])

    gt_df["gt_female"] = 1 - gt_df["gt_male"]

    gt_male_list = gt_df["gt_male"].tolist()
    gt_female_list = gt_df["gt_female"].tolist()

    SAMPLES_PER_OCC = 100

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    #print(tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, torch_dtype="bfloat16", cache_dir="/data/projects/punim1996/anaconda3/.cache/hub")
    if "baseline" not in args.ckpt:
        model = PeftModel.from_pretrained(model, args.ckpt)
    print(args.ckpt)
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
    
    def get_phrase_prob(logits, phrase_ids):
        """
        logits: [vocab] logits vector
        phrase_ids: list[int], e.g., [1234, 5678] for "Female"
        """
        probs = torch.softmax(logits, dim=-1)  # [vocab]
        # 如果 phrase 有多个 tokens -> 取第一个 token 的 prob 作为估计
        return probs[phrase_ids[0]].item()
    
    # Inference
    results = []
    occupation_male_probs = []
    occupation_female_probs = []

    curr_male_probs = []
    curr_female_probs = []

    male_id = tokenizer.encode("Male", add_special_tokens=False)
    female_id = tokenizer.encode("Female", add_special_tokens=False)

    for idx, example in enumerate(tqdm(test_data)):
        prompt = example["messages"][0]["content"]

        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
       #q print(updated_text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        if "GDPO" in args.ckpt:
            messages = [
                {"role": "user", "content": f"<Q>{prompt}<A>"}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        #q print(updated_text)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            # 第一步：生成一个token来获取"0"和"1"的概率
            with torch.no_grad():
                outputs = model(**model_inputs)
                next_token_logits = outputs.logits[0, -1, :]  # 获取最后一个位置的logits

            num_classes = 2
    

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

            # # # 输出各类别概率（可选）
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
            model_inputs = tokenizer([updated_text], return_tensors="pt").to(model.device)
            
        
        # 正常生成剩余的tokens
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8,  # 减少1因为我们已经添加了一个digit
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id = tokenizer.eos_token_id
        )

           # ==== 2️⃣ 拿到每一步的 logits ====
        # generated_ids.scores 是一个 list，每个都是 [1, vocab_size] 的 logits
        # 取生成的第一个 token 的 logits 作为概率（你是做二分类 bias，本质就看第一步 token）
        first_step_logits = generated_ids.scores[1][0]  # Tensor: [vocab]
        
        # ==== 3️⃣ 算 Male / Female 的概率 ====
        male_p = get_phrase_prob(first_step_logits, male_id)
        female_p = get_phrase_prob(first_step_logits, female_id)
        # print(male_p)
        # print(female_p)
        curr_male_probs.append(male_p)
        curr_female_probs.append(female_p)

        if (idx + 1) % SAMPLES_PER_OCC == 0:
            occupation_male_probs.append(np.mean(curr_male_probs))
            occupation_female_probs.append(np.mean(curr_female_probs))
            curr_male_probs = []
            curr_female_probs = []

        # output_ids = generated_ids.sequences
        # generated_token_ids = [
        #     output_ids[i][len(model_inputs.input_ids[i]):] for i in range(len(output_ids))
        # ]
        # # logits_scores = generated_ids.scores
        # print(generated_token_ids)
        # # # 如果需要完整的生成结果（包括采样的digit）
        # response = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]

        # print(response)

    #     results.append({
    #         "index": example["index"],
    #         "messages": example["messages"],
    #         "response": response
    #     })

    # # Optionally save results
    # with open("/data/projects/punim1996/Data/AACL2025/Ours/test.json", "w") as f:
    #     json.dump(results, f, indent=2)

    occupation_mae_male = []
    occupation_mae_female = []

    for occ_id in range(len(occupation_male_probs)):
        model_male = occupation_male_probs[occ_id]
        model_female = occupation_female_probs[occ_id]
        if "even" in args.datasets:
            gt_male = 0.5
            gt_female = 0.5
        else:
            gt_male = gt_male_list[occ_id]
            gt_female = gt_female_list[occ_id]

        occupation_mae_male.append(abs(model_male - gt_male))
        occupation_mae_female.append(abs(model_female - gt_female))


    # =============================
    #  Final Avg MAE
    # =============================
    final_mae_male = np.mean(occupation_mae_male)
    final_mae_female = np.mean(occupation_mae_female)
    final_mae = (final_mae_male + final_mae_female) / 2 
    print("\n====================================")
    print("Final Average MAE across occupations:")
    print(f"MAE:   {final_mae:.6f}")
    print("====================================\n")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    argparser.add_argument('--ckpt', type=str, default='/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_real_uk/Qwen2.5-7B')
    argparser.add_argument('--datasets', type=str, default='/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v2.json')
    
    args = argparser.parse_args()
    #print(args)
    inference(args)

# main function
if __name__ == '__main__':
    main()
        
    