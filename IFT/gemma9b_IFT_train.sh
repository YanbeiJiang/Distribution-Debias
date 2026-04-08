#!/bin/bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate vllm

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model google/gemma-2-9b-it \
    --train_type lora \
    --dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/uk_debias_data_IFT_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_occ_gender_real_uk/gemma-2-9b-it \
    --warmup_ratio 0.05 \
    --use_hf true \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model google/gemma-2-9b-it \
    --train_type lora \
    --dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/uk_debias_data_IFT_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_occ_gender_even_uk/gemma-2-9b-it \
    --warmup_ratio 0.05 \
    --use_hf true \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model google/gemma-2-9b-it \
    --train_type lora \
    --dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_real/us_debias_data_IFT_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_occ_gender_real_us/gemma-2-9b-it \
    --warmup_ratio 0.05 \
    --use_hf true \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model google/gemma-2-9b-it \
    --train_type lora \
    --dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_gender_even/us_debias_data_IFT_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_occ_gender_even_us/gemma-2-9b-it \
    --warmup_ratio 0.05 \
    --use_hf true \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model google/gemma-2-9b-it \
    --train_type lora \
    --dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_race_even/us_race_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_occ_race_us/gemma-2-9b-it \
    --warmup_ratio 0.05 \
    --use_hf true \

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model google/gemma-2-9b-it \
    --train_type lora \
    --dataset '/data/projects/punim1996/Data/AACL2025/dataset_occ_sentiment_even/us_sentiment_train.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_occ_sentiment_us/gemma-2-9b-it \
    --warmup_ratio 0.05 \
    --use_hf true \