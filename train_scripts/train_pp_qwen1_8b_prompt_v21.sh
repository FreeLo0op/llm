# Experimental environment: A100
# 30GB GPU memory
# --tuner_backend swift \
# /mnt/cfs/NLP/lqt/huanhuan-chat-master/dataset/model/ChatGLM2-6B
# /mnt/cfs/NLP/lqt/swift_training/input/mix_328—poem_train_data.json
# /mnt/cfs/NLP/hub_models/Qwen1.5-7B-Chat
# PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_type qwen1half-1_8b-chat \
    --model_cache_dir /mnt/cfs/NLP/hub_models/Qwen1.5-1.8B-Chat \
    --sft_type lora \
    --tuner_backend peft \
    --dtype AUTO \
    --output_dir /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_1_8b_chat_prompt_v21/ \
    --custom_train_dataset_path /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data/g2p_train_prompt_v21.json \
    --train_dataset_sample -1 \
    --num_train_epochs 10 \
    --custom_val_dataset_path /mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/data/g2p_dev_prompt_v21.json \
    --val_dataset_sample -1 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 16 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --self_cognition_sample 1000 \
    --model_name xiaosi \
    --model_author xiaosi
