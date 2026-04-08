#!/bin/bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate vllm

base_output_dir="./output_occ_gender_real_uk/Qwen2.5-7B-Instruct"
if [ -d "$base_output_dir" ]; then
    subfolder1=$(find "$base_output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
    if [ -n "$subfolder1" ]; then
        # Find the checkpoint directory (subfolder2) within subfolder1
        subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -n "$subfolder2" ]; then
            checkpoint_dir="$subfolder2"
        else
            echo "Error: Could not find checkpoint subdirectory in $subfolder1"
            exit 1
        fi
    else
        echo "Error: Could not find version subdirectory in $base_output_dir"
        exit 1
    fi
else
    echo "Error: Base output directory $base_output_dir does not exist"
    exit 1
fi
# occ_gender_real UK
# -------------------------------
result_path_occ_gender_real_uk="./result_occ_gender_real/uk_test_IFT-qwen-7b_v2_ood.jsonl"

if [ -f "$result_path_occ_gender_real_uk" ]; then
    rm "$result_path_occ_gender_real_uk"
    echo "File has been deleted: $result_path_occ_gender_real_uk"
else
    echo "File not existing: $result_path_occ_gender_real_uk"
fi

dataset_occ_gender_real_uk="../dataset_occ_gender_real/us_debias_data_IFT_test_v2.json"

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $checkpoint_dir \
    --infer_backend vllm \
    --val_dataset $dataset_occ_gender_real_uk \
    --temperature 1.0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 16 \
    --result_path $result_path_occ_gender_real_uk


base_output_dir="./output_occ_gender_even_uk/Qwen2.5-7B-Instruct"
if [ -d "$base_output_dir" ]; then
    subfolder1=$(find "$base_output_dir" -maxdepth 1 -type d -name "v*" | sort -V | tail -1)
    if [ -n "$subfolder1" ]; then
        # Find the checkpoint directory (subfolder2) within subfolder1
        subfolder2=$(find "$subfolder1" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -n "$subfolder2" ]; then
            checkpoint_dir="$subfolder2"
        else
            echo "Error: Could not find checkpoint subdirectory in $subfolder1"
            exit 1
        fi
    else
        echo "Error: Could not find version subdirectory in $base_output_dir"
        exit 1
    fi
else
    echo "Error: Base output directory $base_output_dir does not exist"
    exit 1
fi
# occ_gender_even UK
# -------------------------------
result_path_occ_gender_even_uk="./result_occ_gender_even/uk_test_IFT-qwen-7b_v2_ood.jsonl"

if [ -f "$result_path_occ_gender_even_uk" ]; then
    rm "$result_path_occ_gender_even_uk"
    echo "File has been deleted: $result_path_occ_gender_even_uk"
else
    echo "File not existing: $result_path_occ_gender_even_uk"
fi

dataset_occ_gender_even_uk="../dataset_occ_gender_even/us_debias_data_IFT_test_v2.json"

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $checkpoint_dir \
    --infer_backend vllm \
    --val_dataset $dataset_occ_gender_even_uk \
    --temperature 1.0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 16 \
    --result_path $result_path_occ_gender_even_uk



echo "Results for occ gender real uk"
python postprocess_v2_50_50.py --run_name $result_path_occ_gender_real_uk --mode real --location us --attribute gender --version v2

echo "Results for occ gender even uk"
python postprocess_v2_50_50.py --run_name $result_path_occ_gender_even_uk --mode even --location us --attribute gender --version v2

