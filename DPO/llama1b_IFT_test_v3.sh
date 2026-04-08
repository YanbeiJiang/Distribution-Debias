#!/bin/bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate vllm

base_output_dir="./output_occ_gender_real_uk/Llama-3.2-1B-Instruct"
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
result_path_occ_gender_real_uk="./result_occ_gender_real/uk_test_IFT-llama-1b_v3.jsonl"

if [ -f "$result_path_occ_gender_real_uk" ]; then
    rm "$result_path_occ_gender_real_uk"
    echo "File has been deleted: $result_path_occ_gender_real_uk"
else
    echo "File not existing: $result_path_occ_gender_real_uk"
fi

dataset_occ_gender_real_uk="../dataset_occ_gender_real/uk_debias_data_IFT_test_v3.json"

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


base_output_dir="./output_occ_gender_even_uk/Llama-3.2-1B-Instruct"
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
result_path_occ_gender_even_uk="./result_occ_gender_even/uk_test_IFT-llama-1b_v3.jsonl"

if [ -f "$result_path_occ_gender_even_uk" ]; then
    rm "$result_path_occ_gender_even_uk"
    echo "File has been deleted: $result_path_occ_gender_even_uk"
else
    echo "File not existing: $result_path_occ_gender_even_uk"
fi

dataset_occ_gender_even_uk="../dataset_occ_gender_even/uk_debias_data_IFT_test_v3.json"

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

base_output_dir="./output_occ_gender_real_us/Llama-3.2-1B-Instruct"
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
# occ_gender_real US
# -------------------------------
result_path_occ_gender_real_us=./result_occ_gender_real/us_test_IFT-llama-1b_v3.jsonl

if [ -f "$result_path_occ_gender_real_us" ]; then
    rm "$result_path_occ_gender_real_us"
    echo "File has been deleted: $result_path_occ_gender_real_us"
else
    echo "File not existing: $result_path_occ_gender_real_us"
fi

dataset_occ_gender_real_us="../dataset_occ_gender_real/us_debias_data_IFT_test_v3.json"

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $checkpoint_dir \
    --infer_backend vllm \
    --val_dataset $dataset_occ_gender_real_us \
    --temperature 1.0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 16 \
    --result_path $result_path_occ_gender_real_us

base_output_dir="./output_occ_gender_even_us/Llama-3.2-1B-Instruct"
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
# occ_gender_even US
# -------------------------------
result_path_occ_gender_even_us="./result_occ_gender_even/us_test_IFT-llama-1b_v3.jsonl"

if [ -f "$result_path_occ_gender_even_us" ]; then
    rm "$result_path_occ_gender_even_us"
    echo "File has been deleted: $result_path_occ_gender_even_us"
else
    echo "File not existing: $result_path_occ_gender_even_us"
fi

dataset_occ_gender_even_us="../dataset_occ_gender_even/us_debias_data_IFT_test_v3.json"

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $checkpoint_dir \
    --infer_backend vllm \
    --val_dataset $dataset_occ_gender_even_us \
    --temperature 1.0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 16 \
    --result_path $result_path_occ_gender_even_us

base_output_dir="./output_occ_sentiment_us/Llama-3.2-1B-Instruct"
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
# occ_race_sentiment
# -------------------------------
result_path_occ_sentiment_even="./result_occ_sentiment_even/us_test_IFT-llama-1b_v3.jsonl"

if [ -f "$result_path_occ_sentiment_even" ]; then
    rm "$result_path_occ_sentiment_even"
    echo "File has been deleted: $result_path_occ_sentiment_even"
else
    echo "File not existing: $result_path_occ_sentiment_even"
fi

dataset_occ_sentiment_even="../dataset_occ_sentiment_even/us_sentiment_test_v3.json"

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $checkpoint_dir \
    --infer_backend vllm \
    --val_dataset $dataset_occ_sentiment_even \
    --temperature 1.0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --max_batch_size 16 \
    --result_path $result_path_occ_sentiment_even

# ./start_vllm
# sleep 120

echo "Results for occ gender real uk"
python postprocess_v2_50_50.py --run_name $result_path_occ_gender_real_uk --mode real --location uk --attribute gender --version v3

echo "Results for occ gender even uk"
python postprocess_v2_50_50.py --run_name $result_path_occ_gender_even_uk --mode even --location uk --attribute gender --version v3

echo "Results for occ gender real us"
python postprocess_v2_50_50.py --run_name $result_path_occ_gender_real_us --mode real --location us --attribute gender --version v3

echo "Results for occ gender even us"
python postprocess_v2_50_50.py --run_name $result_path_occ_gender_even_us --mode even --location us --attribute gender --version v3

echo "Results for occ sentiment even"
python postprocess_v2_50_50.py --run_name $result_path_occ_sentiment_even --mode even --location us --attribute sentiment --version v3