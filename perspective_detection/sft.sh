#!/bin/bash
for warmup_ratio in 0.1
do
    for weight_decay in 0.001
    do
        for sch in "linear"
        do
            for lr in 1e-4
            do
                for epoch in 2
                do
                    echo "save to saved_models/Mistral_test_ep${epoch}_lr${lr}_wc${weight_decay}_wr${warmup_ratio}_sch${sch}"
                    accelerate launch --main_process_port 29500 train_sft.py \
                                --output_dir "./results" \
                                --num_train_epochs ${epoch} \
                                --per_device_train_batch_size 1 \
                                --gradient_checkpointing True \
                                --gradient_accumulation_steps 16 \
                                --per_device_eval_batch_size 1 \
                                --optim "adamw_torch" \
                                --save_steps 5000 \
                                --logging_steps 10 \
                                --learning_rate ${lr} \
                                --weight_decay ${weight_decay} \
                                --fp16 True \
                                --bf16 False \
                                --max_grad_norm 1.0 \
                                --max_steps -1 \
                                --warmup_ratio ${warmup_ratio} \
                                --group_by_length True \
                                --lr_scheduler_type ${sch} \
                                --dataloader_num_workers 4 \
                                --dataloader_prefetch_factor 2 \
                                --evaluation_strategy "steps" \
                                --eval_accumulation_steps 16 \
                                --load_best_model_at_end \
                                --save_model "saved_models/Mistral_test_ep${epoch}_lr${lr}_wc${weight_decay}_wr${warmup_ratio}_sch${sch}"  
                done               
            done
        done
    done
done
