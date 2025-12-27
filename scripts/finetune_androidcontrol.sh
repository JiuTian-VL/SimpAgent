export PYTHONPATH=src:$PYTHONPATH

MODEL_NAME="/data1/Models/Qwen2-VL-2B-Instruct"


deepspeed --include localhost:4,5,6,7 --master_port 29473 src/training/training_augment_distill_with_alpha_with_drop_highres.py \
    --lora_enable True \
    --lora_namespan_exclude "['model.embed_tokens', 'lm_head']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path "/data7/Users/zxr/hyh/SimpAgent-main/data/androidcontrol_train_llavaformat.json" \
    --image_folder "" \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "/data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/outputs/androidcontrol_bs128_lr3e-4_mask535_drop6_lora" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --min_pixels $((256 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 3e-4 \
    --merger_lr 0.0 \
    --vision_lr 0.0 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000  \
    --save_total_limit 11 \
    --seed 42 \
    --save_only_model True \
    --dataloader_num_workers 12 \
    --alpha 1 \
    --drop_k 6 \
    --augment True \
    --mask_prob 0.5 \
    --window_min 0.3 \
    --window_max 0.5 \
    --change True \
    --combination False \
    --random_mask True   


CUDA_VISIBLE_DEVICES="0" python3 src/merge_lora_weights.py \
    --model-path /data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/outputs/androidcontrol_bs128_lr3e-4_mask535_drop6_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/Models/androidcontrol_bs128_lr3e-4_mask535_drop6_lora \
    --safe-serialization