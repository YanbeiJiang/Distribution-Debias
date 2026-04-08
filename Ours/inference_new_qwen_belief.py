import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import os

def inference(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        torch_dtype="bfloat16",
        cache_dir="/data/projects/punim1996/anaconda3/.cache/hub"
    )
    model = PeftModel.from_pretrained(model, args.ckpt)
    model.cuda()
    model.eval()

    with open(args.datasets, "r") as f:
        test_data = json.load(f)

    OCC_SIZE = 100

    occ_belief_male = []       # 每个 occupation 的 mean belief male prob
    occ_generated_male = []    # 每个 occupation 的 mean generated male prob

    # 遍历每个 occupation
    for occ_start in tqdm(range(0, len(test_data), OCC_SIZE), desc="Processing occupations"):
        occ_slice = test_data[occ_start: occ_start + OCC_SIZE]

        belief_probs = []
        generated_probs = []

        for example in occ_slice:

            prompt = example["messages"][0]["content"]

            # step1: belief token
            messages = [
                {"role": "user", "content": f"<Q>{prompt}<A>"}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**model_inputs)
                logits = outputs.logits[0, -1, :]

            # belief classes: 0=male, 1=female
            token_ids = [
                tokenizer.encode("0", add_special_tokens=False)[0],
                tokenizer.encode("1", add_special_tokens=False)[0]
            ]
            relevant_logits = torch.stack([logits[i] for i in token_ids])
            probs = F.softmax(relevant_logits, dim=0)

            prob_male = probs[0].item()
            prob_female = probs[1].item()
            belief_probs.append(prob_male)

            # step2: sample digit and generate gender
            sampled_digit = "0" if np.random.rand() < prob_male else "1"

            updated_messages = [
                {"role": "user", "content": f"<Q>{prompt}<A>{sampled_digit}"}
            ]
            updated_text = tokenizer.apply_chat_template(
                updated_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            updated_inputs = tokenizer([updated_text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **updated_inputs,
                max_new_tokens=20,
                return_dict_in_generate=True,
            )

            output_ids = generated_ids.sequences[0][len(updated_inputs.input_ids[0]):]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            #print(response)
            # extract gender
            resp = response.strip().lower().replace("0","").replace("1","").replace("2","").replace("3","").replace("4","")
            if "male" == resp:
                generated_probs.append(1.0)
            elif "female" == resp:
                generated_probs.append(0.0)
            else:
                # fallback
                generated_probs.append(prob_male)

        # occupation-level averages
        occ_belief_male.append(np.mean(belief_probs))
        occ_generated_male.append(np.mean(generated_probs))

    # ---------- Save scatter plot ----------
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    plt.figure(figsize=(6, 6))

    plt.scatter(occ_belief_male, occ_generated_male)
    plt.plot([0, 1], [0, 1], 'r--')  # ideal line

    # Labels & title (bigger fonts)
    plt.xlabel("Steering Token Male Probability", fontsize=16)
    plt.ylabel("Generated Male Probability", fontsize=16)
    plt.title("Steering vs Generated Gender", fontsize=18)

    # Tick numbers
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True)

    plt.savefig(
        os.path.join(os.path.dirname(args.output_dir), "scatter_plot_real_uk_qwen_1.5b.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ---------- Save CSV ----------
    save_csv = os.path.join(os.path.dirname(args.output_dir), "belief_vs_generated_real_uk_qwen_1.5b.csv")
    with open(save_csv, "w") as f:
        f.write("occ_idx,belief_male,generated_male\n")
        for i, (b, g) in enumerate(zip(occ_belief_male, occ_generated_male)):
            f.write(f"{i},{b},{g}\n")

    print("Saved scatter plot and CSV!")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    argparser.add_argument('--ckpt', type=str, default='/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_real_uk/Qwen2.5-1.5B')
    argparser.add_argument('--output_dir', type=str, default='/data/projects/punim1996/Data/AACL2025/Ours/outputs_belief_prob')
    argparser.add_argument('--datasets', type=str, default='/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v2.json')
    args = argparser.parse_args()

    inference(args)

if __name__ == '__main__':
    main()
