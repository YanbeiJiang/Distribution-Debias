#!/bin/bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate gender_debias

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
for pid in $PIDS; do
    echo "Killing process $pid"
    kill -9 $pid
done

# dataset_occ_gender_real uk train
# --------------------------
model="google/gemma-2-2b-it"
output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_real_uk/gemma2-2b
rm -rf "${output_dir}"
CUDA_VISIBLE_DEVICES=0,1 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_gdpo_train_single_response.csv] loss=sft exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=4 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=2 output_dir=$output_dir seed=42 lr=1e-4

CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it model.name_or_path=$output_dir datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_gdpo_train_single_response.csv] loss=kto loss.beta=0.1 exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=8 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=3 output_dir=$output_dir seed=42 lr=2e-6
# conda activate vllm
# python /data/projects/punim1996/Data/AACL2025/Ours/save_model.py --model $model --output_dir $output_dir
# 
# dataset_occ_gender_real uk test
# --------------------------
#model_dir="${output_dir}"

output_dir_occ_gender_real_uk_v1="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/uk_test_GDPO-gemma-2b_v1.json"
dataset_v1="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v1.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_real_uk_v1 --datasets $dataset_v1

output_dir_occ_gender_real_uk_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/uk_test_GDPO-gemma-2b_v2.json"
dataset_v2="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v2.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_real_uk_v2 --datasets $dataset_v2

output_dir_occ_gender_real_uk_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/uk_test_GDPO-gemma-2b_v3.json"
dataset_v3="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v3.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_real_uk_v3 --datasets $dataset_v3
# conda activate vllm
# output_dir_occ_gender_real_uk_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/uk_test_GDPO-gemma-2b_v3.jsonl"

# if [ -f "$output_dir_occ_gender_real_uk_v3" ]; then
#     rm "$output_dir_occ_gender_real_uk_v3"
#     echo "Deleted: $output_dir_occ_gender_real_uk_v3"
# else
#     echo "Not exist: $output_dir_occ_gender_real_uk_v3"
# fi

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model google/gemma-2-2b-it \
#     --adapters $output_dir \
#     --val_dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_test_v3.json' \
#     --infer_backend vllm \
#     --temperature 1.0 \
#     --max_new_tokens 1024 \
#     --use_hf true \
#     --max_batch_size 1 \
#     --result_path $output_dir_occ_gender_real_uk_v3
    

# dataset_occ_gender_real us train
# --------------------------
model="google/gemma-2-2b-it"
output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_real_us/gemma2-2b
rm -rf "${output_dir}"
CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_gdpo_train_single_response.csv] loss=sft exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=4 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=2 output_dir=$output_dir seed=42 lr=1e-4 model.r=8

CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it model.name_or_path=$output_dir datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_gdpo_train_single_response.csv] loss=kto loss.beta=0.1 exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=16 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=3 output_dir=$output_dir seed=42 lr=1e-6 model.r=8
# conda activate vllm
# python /data/projects/punim1996/Data/AACL2025/Ours/save_model.py --model $model --output_dir $output_dir
# 
# # dataset_occ_gender_real us test
# # --------------------------
# model_dir="${output_dir}/policy.pt"

output_dir_occ_gender_real_us_v1="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/us_test_GDPO-gemma-2b_v1.json"
dataset_v1="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_IFT_test_v1.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_real_us_v1 --datasets $dataset_v1

output_dir_occ_gender_real_us_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/us_test_GDPO-gemma-2b_v2.json"
dataset_v2="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_IFT_test_v2.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_real_us_v2 --datasets $dataset_v2

output_dir_occ_gender_real_us_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/us_test_GDPO-gemma-2b_v3.json"
dataset_v3="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_IFT_test_v3.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_real_us_v3 --datasets $dataset_v3
# conda activate vllm
# output_dir_occ_gender_real_us_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_real/us_test_GDPO-gemma-2b_v3.jsonl"

# if [ -f "$output_dir_occ_gender_real_us_v3" ]; then
#     rm "$output_dir_occ_gender_real_us_v3"
#     echo "Deleted: $output_dir_occ_gender_real_us_v3"
# else
#     echo "Not exist: $output_dir_occ_gender_real_us_v3"
# fi

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model $output_dir \
#     --val_dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_IFT_test_v3.json' \
#     --infer_backend vllm \
#     --temperature 1.0 \
#     --max_new_tokens 1024 \
#     --use_hf true \
#     --max_batch_size 1 \
#     --result_path $output_dir_occ_gender_real_us_v3



# dataset_occ_gender_even uk train
# --------------------------
model="google/gemma-2-2b-it"
output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_even_uk/gemma2-2b
rm -rf "${output_dir}"
CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_gdpo_train_single_response.csv] loss=sft exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=4 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=2 output_dir=$output_dir seed=42 lr=5e-6 model.r=8

CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it model.name_or_path=$output_dir datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_gdpo_train_single_response.csv] loss=kto loss.beta=0.1 exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=16 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=3 output_dir=$output_dir seed=42 lr=1e-6 model.r=8
# conda activate vllm
# python /data/projects/punim1996/Data/AACL2025/Ours/save_model.py --model $model --output_dir $output_dir
# 
# # dataset_occ_gender_even uk test
# # --------------------------
# model_dir="${output_dir}/policy.pt"

output_dir_occ_gender_even_uk_v1="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/uk_test_GDPO-gemma-2b_v1.json"
dataset_v1="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_IFT_test_v1.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_even_uk_v1 --datasets $dataset_v1

output_dir_occ_gender_even_uk_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/uk_test_GDPO-gemma-2b_v2.json"
dataset_v2="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_IFT_test_v2.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_even_uk_v2 --datasets $dataset_v2

output_dir_occ_gender_even_uk_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/uk_test_GDPO-gemma-2b_v3.json"
dataset_v3="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_IFT_test_v3.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_even_uk_v3 --datasets $dataset_v3
# conda activate vllm
# output_dir_occ_gender_even_uk_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/uk_test_GDPO-gemma-2b_v3.jsonl"

# if [ -f "$output_dir_occ_gender_even_uk_v3" ]; then
#     rm "$output_dir_occ_gender_even_uk_v3"
#     echo "Deleted: $output_dir_occ_gender_even_uk_v3"
# else
#     echo "Not exist: $output_dir_occ_gender_even_uk_v3"
# fi

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model $output_dir \
#     --val_dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_IFT_test_v3.json' \
#     --infer_backend vllm \
#     --temperature 1.0 \
#     --max_new_tokens 1024 \
#     --use_hf true \
#     --max_batch_size 1 \
#     --result_path $output_dir_occ_gender_even_uk_v3


# dataset_occ_gender_even us train
# --------------------------
model="google/gemma-2-2b-it"
output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_gender_even_us/gemma2-2b
rm -rf "${output_dir}"
CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_gdpo_train_single_response.csv] loss=sft exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=4 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=2 output_dir=$output_dir seed=42 lr=1e-4 model.r=32

CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it model.name_or_path=$output_dir datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_gdpo_train_single_response.csv] loss=kto loss.beta=0.1 exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=8 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=3 output_dir=$output_dir seed=42 lr=1e-6 model.r=32
# conda activate vllm
# python /data/projects/punim1996/Data/AACL2025/Ours/save_model.py --model $model --output_dir $output_dir
# 
# # dataset_occ_gender_even us test
# # --------------------------
# model_dir="${output_dir}/policy.pt"

output_dir_occ_gender_even_us_v1="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/us_test_GDPO-gemma-2b_v1.json"
dataset_v1="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_IFT_test_v1.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_even_us_v1 --datasets $dataset_v1

output_dir_occ_gender_even_us_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/us_test_GDPO-gemma-2b_v2.json"
dataset_v2="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_IFT_test_v2.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_even_us_v2 --datasets $dataset_v2

output_dir_occ_gender_even_us_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/us_test_GDPO-gemma-2b_v3.json"
dataset_v3="/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_IFT_test_v3.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_gender_even_us_v3 --datasets $dataset_v3
# conda activate vllm
# output_dir_occ_gender_even_us_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_gender_even/us_test_GDPO-gemma-2b_v3.jsonl"

# if [ -f "$output_dir_occ_gender_even_us_v3" ]; then
#     rm "$output_dir_occ_gender_even_us_v3"
#     echo "Deleted: $output_dir_occ_gender_even_us_v3"
# else
#     echo "Not exist: $output_dir_occ_gender_even_us_v3"
# fi

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model $output_dir \
#     --val_dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_IFT_test_v3.json' \
#     --infer_backend vllm \
#     --temperature 1.0 \
#     --max_new_tokens 1024 \
#     --use_hf true \
#     --max_batch_size 1 \
#     --result_path $output_dir_occ_gender_even_us_v3


# dataset_occ_race_even us train
# --------------------------
model="google/gemma-2-2b-it"
output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_race_even_us/gemma2-2b
rm -rf "${output_dir}"
CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_race_even/us_debias_data_gdpo_train_single_response.csv] loss=sft exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=4 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=2 output_dir=$output_dir seed=42 lr=5e-4 model.r=8

CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it model.name_or_path=$output_dir datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_race_even/us_debias_data_gdpo_train_single_response.csv] loss=kto loss.beta=0.1 exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=16 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=3 output_dir=$output_dir seed=42 lr=1e-6 model.r=8
# conda activate vllm
# python /data/projects/punim1996/Data/AACL2025/Ours/save_model.py --model $model --output_dir $output_dir
# 
# # dataset_occ_race_even us test
# # --------------------------
# model_dir="${output_dir}/policy.pt"
output_dir_occ_race_even_us_v1="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_race_even/us_test_GDPO-gemma-2b_v1.json"
dataset_v1="/data/projects/punim1996/Data/AACL2025/dataset_occ_race_even/us_race_test_v1.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_race_even_us_v1 --datasets $dataset_v1

output_dir_occ_race_even_us_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_race_even/us_test_GDPO-gemma-2b_v2.json"
dataset_v2="/data/projects/punim1996/Data/AACL2025/dataset_occ_race_even/us_race_test_v2.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_race_even_us_v2 --datasets $dataset_v2


# dataset_occ_sentiment_even us train
# --------------------------
model="google/gemma-2-2b-it"
output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_sentiment_even_us/gemma2-2b
rm -rf "${output_dir}"
CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_debias_data_gdpo_train_single_response.csv] loss=sft exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=4 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=2 output_dir=$output_dir seed=42 lr=1e-4 model.r=64

CUDA_VISIBLE_DEVICES=0 python -u /data/projects/punim1996/Data/AACL2025/Ours/train.py model=Gemma2-2B-it model.name_or_path=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_sentiment_even_us/gemma2-2b datasets=[/data/gpfs/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_debias_data_gdpo_train_single_response.csv] loss=kto loss.beta=0.1 exp_name=uk-gemma2-2b gradient_accumulation_steps=1 batch_size=16 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 eval_every=1000000 n_epochs=3 output_dir=/data/projects/punim1996/Data/AACL2025/Ours/output_occ_sentiment_even_us/gemma2-2b seed=42 lr=1e-6 model.r=64
# conda activate vllm
# python /data/projects/punim1996/Data/AACL2025/Ours/save_model.py --model $model --output_dir $output_dir
# 
# # dataset_occ_sentiment_even us test
# # --------------------------
# model_dir="${output_dir}/policy.pt"

output_dir_occ_sentiment_even_us_v1="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_sentiment_even/us_test_GDPO-gemma-2b_v1.json"
dataset_v1="/data/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_sentiment_test_v1.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_sentiment_even_us_v1 --datasets $dataset_v1

output_dir_occ_sentiment_even_us_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_sentiment_even/us_test_GDPO-gemma-2b_v2.json"
dataset_v2="/data/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_sentiment_test_v2.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_sentiment_even_us_v2 --datasets $dataset_v2

output_dir_occ_sentiment_even_us_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_sentiment_even/us_test_GDPO-gemma-2b_v3.json"
dataset_v3="/data/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_sentiment_test_v3.json"

CUDA_VISIBLE_DEVICES=0 python /data/projects/punim1996/Data/AACL2025/Ours/inference_new_qwen.py --model $model --ckpt $output_dir --output_dir $output_dir_occ_sentiment_even_us_v3 --datasets $dataset_v3

# output_dir_occ_sentiment_even_us_v2="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_sentiment_even/us_test_GDPO-gemma-2b_v2.jsonl"

# if [ -f "$output_dir_occ_sentiment_even_us_v2" ]; then
#     rm "$output_dir_occ_sentiment_even_us_v2"
#     echo "Deleted: $output_dir_occ_sentiment_even_us_v2"
# else
#     echo "Not exist: $output_dir_occ_sentiment_even_us_v2"
# fi

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model google/gemma-2-2b-it \
#     --adapters $output_dir \
#     --val_dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_sentiment_test_v2.json' \
#     --infer_backend vllm \
#     --temperature 1.0 \
#     --max_new_tokens 1024 \
#     --use_hf true \
#     --max_batch_size 1 \
#     --result_path $output_dir_occ_sentiment_even_us_v2


# conda activate vllm
# output_dir_occ_sentiment_even_us_v3="/data/projects/punim1996/Data/AACL2025/Ours/result_occ_sentiment_even/us_test_GDPO-gemma-2b_v3.jsonl"

# if [ -f "$output_dir_occ_sentiment_even_us_v3" ]; then
#     rm "$output_dir_occ_sentiment_even_us_v3"
#     echo "Deleted: $output_dir_occ_sentiment_even_us_v3"
# else
#     echo "Not exist: $output_dir_occ_sentiment_even_us_v3"
# fi

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --model google/gemma-2-2b-it \
#     --adapters $output_dir \
#     --val_dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_sentiment_test_v3.json' \
#     --infer_backend vllm \
#     --temperature 1.0 \
#     --max_new_tokens 1024 \
#     --use_hf true \
#     --max_batch_size 1 \
#     --result_path $output_dir_occ_sentiment_even_us_v3

# Evaluation Results
# ---------------------
/data/projects/punim1996/Data/AACL2025/Ours/start_vllm.sh
sleep 180


echo "V1 Results for occ gender real uk"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_real_uk_v1 --mode real --location uk --attribute gender --version v1

echo "V2 Results for occ gender real uk"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_real_uk_v2 --mode real --location uk --attribute gender --version v2

echo "V3 Results for occ gender real uk"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_real_uk_v3 --mode real --location uk --attribute gender --version v3

echo "V1 Results for occ gender even uk"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_even_uk_v1 --mode even --location uk --attribute gender --version v1

echo "V2 Results for occ gender even uk"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_even_uk_v2 --mode even --location uk --attribute gender --version v2

echo "V3 Results for occ gender even uk"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_even_uk_v3 --mode even --location uk --attribute gender --version v3

echo "V1 Results for occ gender real us"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_real_us_v1 --mode real --location us --attribute gender --version v1

echo "V2 Results for occ gender real us"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_real_us_v2 --mode real --location us --attribute gender --version v2

echo "V3 Results for occ gender real us"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_real_us_v3 --mode real --location us --attribute gender --version v3

echo "V1 Results for occ gender even us"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_even_us_v1 --mode even --location us --attribute gender --version v1

echo "V2 Results for occ gender even us"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_even_us_v2 --mode even --location us --attribute gender --version v2

echo "V3 Results for occ gender even us"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_gender_even_us_v3 --mode even --location us --attribute gender --version v3

echo "V1 Results for occ race even"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_race_even_us_v1 --mode even --location us --attribute race --version v1

echo "V2 Results for occ race even"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_race_even_us_v2 --mode even --location us --attribute race --version v2

echo "V1 Results for occ sentiment even"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_sentiment_even_us_v1 --mode even --location us --attribute sentiment --version v1

echo "V2 Results for occ sentiment even"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_sentiment_even_us_v2 --mode even --location us --attribute sentiment --version v2

echo "V3 Results for occ sentiment even"
python /data/projects/punim1996/Data/AACL2025/Ours/postprocess_v2_50_50.py --run_name $output_dir_occ_sentiment_even_us_v3 --mode even --location us --attribute sentiment --version v3