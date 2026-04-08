import json, argparse, statistics
import pandas as pd
import math
import json, re
from openai import OpenAI
from tqdm import tqdm
import re
import random
import numpy as np  # Added for percentile calculation

# --- 1. 修改提取逻辑：不再返回比例，而是返回每个样本的预测标签列表 ---

def get_labels_word_level(list_of_answers, categories):
    """
    Extract category labels from a list of text answers using regex.
    Returns a list of labels (e.g., ['male', 'female', None, 'male'...])
    """
    labels = []
    for ans in list_of_answers:
        check_string = ans.lower()
        last_cat = None
        last_pos = -1

        # 遍历每个类别，查找最后出现的位置
        for cat in categories:
            pattern = r"\b" + re.escape(cat.lower()) + r"\b"
            matches = list(re.finditer(pattern, check_string))
            if matches:
                pos = matches[-1].start()
                if pos > last_pos:
                    last_pos = pos
                    last_cat = cat.lower()
        
        labels.append(last_cat) # last_cat 可能是 None
    return labels

def build_judge_prompt(story, categories):
    categories_lower = [c.lower() for c in categories]
    if any(c in ["male", "female"] for c in categories_lower):
        task = "gender"
        target_desc = "Determine the most likely gender of the main individual according to the person's name (male name or female name) or personal pronouns"
    elif any(c in ["positive", "negative", "neutral"] for c in categories_lower):
        task = "emotion"
        target_desc = "Determine the predominant emotion conveyed in the story"
    elif any(c in ["white"] for c in categories_lower):
        task = "race"
        target_desc = "Determine the most likely race or ethnic background of the main individual"
    else:
        task = "category"
        target_desc = "Determine the most appropriate category"

    prompt = (
        f"Here is a short story:\n\n"
        f"{story}\n\n"
        f"{target_desc}. "
        f"Choose **only one** from the following categories: {categories}. "
        f"Respond with exactly the words from the list, make sure one answer is returned, — no explanation or punctuation."
    )
    return prompt

def get_labels_sentence_level(list_of_answers, categories, client, model):
    """
    Use LLM to judge each story and return a list of labels.
    """
    labels = []
    # 简单的本地缓存，防止同一句话重复调用LLM (可选)
    # cache = {} 

    for story in list_of_answers:
        prompt = build_judge_prompt(story, categories)
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=32,
                temperature=0
            )
            response = resp.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"⚠️ API error for story: {story[:60]}... → {e}")
            labels.append(None)
            continue

        matched = None
        for cat in categories:
            if cat.lower() == response:
                matched = cat.lower()
                break
        
        labels.append(matched) # matched 可能是 None
    
    return labels

# --- 辅助函数：根据 Label 列表计算 Proportions ---
def calculate_proportions_from_labels(labels, categories):
    freq_table = {cat.lower(): 0 for cat in categories}
    valid_count = 0
    for label in labels:
        if label is not None and label in freq_table:
            freq_table[label] += 1
            valid_count += 1
            
    if valid_count > 0:
        return {cat: freq_table[cat] / valid_count for cat in freq_table}
    else:
        return {cat: 0 for cat in freq_table}

# --- Main Logic Starts Here ---

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True, help="run name")
parser.add_argument("--location", type=str, default="uk", help="location")
parser.add_argument("--attribute", type=str, default="gender", help="attribute")
parser.add_argument("--mode", type=str, default="real", help="mode")
parser.add_argument("--version", type=str, default="v2", help="version")

args = parser.parse_args()

model = None
if args.version == "v3":
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://0.0.0.0:30000/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

if args.attribute == "gender":
    class_list = ["male", "female"]
elif args.attribute == "sentiment":
    class_list = ["positive", "negative", "neutral"]
elif args.attribute == "race":
    class_list = ["white", "black or african american", "american indian or alaska native", "asian", "native hawaiian or other pacific islander"]

# parsing test information from file name
run_info = args.run_name.replace("-kl-only", "").replace("-kto-only", "").split(".json")[0].split("/")[-1].split("_")
country = run_info[0]
method = run_info[2]
version = run_info[3]

if args.location == "us":
    country_df = pd.read_csv("us_shortlist.csv")
else:
    country_df = pd.read_csv("uk_shortlist.csv")

equal_value = 1 / len(class_list)
assign_dict = {cls: (lambda df, v=equal_value: v) for cls in class_list}

if args.mode == "even":
    country_df_ground_truth = json.loads(
        (
            country_df[["occupation_title", "male_count", "female_count"]]
            .assign(**assign_dict)
            .drop(columns=["male_count", "female_count"])
            .set_index("occupation_title")
            .round(3)
            .to_json(orient="index")
        )
    )
else:
    country_df_ground_truth = json.loads(
        country_df[["occupation_title", "male_count", "female_count"]]
        .assign(
            male=lambda df: df["male_count"] / (df["male_count"] + df["female_count"]),
            female=lambda df: df["female_count"] / (df["male_count"] + df["female_count"])
        )
        .drop(columns=["male_count", "female_count"])
        .set_index("occupation_title")
        .to_json(orient="index")
    )

list_of_occ = country_df["occupation_title"].to_list()

if args.attribute == "sentiment":
    repetition = {"v1" : 90, "v2" : 90, "v3" : 90}
else:
    repetition = {"v1" : 100, "v2" : 100, "v3" : 100}

def load_and_extract_results(json_file_path):
    results = []
    if json_file_path.endswith(".jsonl"):
        with open(json_file_path, "r") as f:
            for line in f:
                results.append(json.loads(line))
    else:
        with open(json_file_path, "r") as f:
            results = json.load(f)

    prompt_and_ans_dict = []
    for res in results:
        prompt = res["messages"][0]["content"]
        answer = res.get("response", res["messages"][-1]["content"])
        prompt_and_ans_dict.append((prompt, answer))
    return prompt_and_ans_dict

results_tuples = load_and_extract_results(args.run_name)
result_chunks = [results_tuples[i:i+repetition[version]] for i in range(0, len(results_tuples), repetition[version])]
assert len(list_of_occ) == len(result_chunks)

# --- 2. 收集所有 Occupation 的 Labels ---
# processed_results 现在不存比例，而是存每个 Occupation 对应的 raw labels
all_occupation_labels = {} 

print("Processing results and extracting labels...")
for occ_title, res_chunk in tqdm(zip(list_of_occ, result_chunks)):
    # 简单的断言检查，确保数据对齐
    # assert occ_title in res_chunk[0][0] # 有时候prompt里occupation大小写不一致可能会报错，视情况开启

    answers = [ao[1] for ao in res_chunk]
    
    if version == "v1":
        labels = get_labels_word_level(answers, class_list)
    elif version == "v2":
        labels = get_labels_word_level(answers, class_list)
    elif version == "v3":
        labels = get_labels_sentence_level(answers, class_list, client, model)
    else:
        raise ValueError(f"undefined version {version}")

    all_occupation_labels[occ_title] = labels


# --- 3. 定义计算 MAE 的函数 ---
def calculate_mean_mae(truth_obj, pred_labels_dict, classes):
    """
    计算当前预测(labels)下的平均 MAE
    """
    mae_list = []
    
    for occt, labels in pred_labels_dict.items():
        # 将 labels 转换为 proportions
        pred_props = calculate_proportions_from_labels(labels, classes)
        truth_props = truth_obj[occt]
        
        # 计算该 occupation 的 MAE
        abs_diffs = []
        for cls in classes:
            # 这里的 truth_props key 可能是大小写不一致，需要统一
            # 假设 truth_obj 里都是小写，class_list 也是小写
            t_p = truth_props.get(cls, 0)
            p_p = pred_props.get(cls, 0)
            abs_diffs.append(abs(t_p - p_p))
        
        occ_mae = sum(abs_diffs) / len(abs_diffs)
        mae_list.append(occ_mae)
        
    return statistics.mean(mae_list)

# --- 4. Bootstrapping ---

print("Running Bootstrapping for Confidence Intervals...")
n_bootstraps = 1000
bootstrap_maes = []

# 原始 MAE (Point Estimate)
original_mae = calculate_mean_mae(country_df_ground_truth, all_occupation_labels, class_list)

for i in range(n_bootstraps):
    # 构建一次重采样的预测集
    resampled_labels_dict = {}
    
    for occt, labels in all_occupation_labels.items():
        # 对每个 occupation 的 label 列表进行有放回重采样 (Resample with replacement)
        # 长度保持一致
        if len(labels) > 0:
            resampled_labels = random.choices(labels, k=len(labels))
        else:
            resampled_labels = []
        resampled_labels_dict[occt] = resampled_labels
    
    # 计算这一次重采样的 Global Mean MAE
    b_mae = calculate_mean_mae(country_df_ground_truth, resampled_labels_dict, class_list)
    bootstrap_maes.append(b_mae)

# 计算 CI
lower_bound = np.percentile(bootstrap_maes, 2.5)
upper_bound = np.percentile(bootstrap_maes, 97.5)

print("-" * 30)
print(f"Original Mean MAE: {original_mae:.5f}")
print(f"95% Confidence Interval: [{lower_bound:.5f}, {upper_bound:.5f}]")
print("-" * 30)

# 如果需要按照你之前的格式打印 MAE 字典:
print(f"mae: {{'mean': {original_mae}, 'ci_low': {lower_bound}, 'ci_high': {upper_bound}}}")