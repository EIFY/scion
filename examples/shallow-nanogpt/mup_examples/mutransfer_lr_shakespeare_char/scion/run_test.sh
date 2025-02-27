for width in 2048 #512 1024 2048
do
    for lr in 0.0078125
    do
        for seed in 2
        do
            head_size=64
            n_heads=$((width / head_size))
            out_dir="mup_examples/mutransfer_lr_shakespeare_char/scion/out/test"
            python train.py \
                --out_dir=$out_dir \
                --log_interval=1 \
                --eval_on_end=True \
                --eval_iters=200 \
                --skip_val_loss=False \
                --eval_only=False \
                --always_save_checkpoint=False \
                --never_save_checkpoint=True \
                --init_from='scratch' \
                --wandb_log=False \
                --csv_log=True \
                --dataset='shakespeare_char' \
                --gradient_accumulation_steps=1 \
                --batch_size=32 \
                --block_size=1024 \
                --n_layer=2 \
                --n_head=$n_heads \
                --n_embd=$width \
                --dropout=0.0 \
                --bias=False \
                --init_std=0.02 \
                --learning_rate=$lr \
                --max_iters=122 \
                --beta1=0.9 \
                --grad_clip=0.0 \
                --warmup_iters=0 \
                --decay_lr=True \
                --scion_enabled=True \
                --scion_first_layer='Sign' \
                --scion_unconstrained=True \
                --seed=$seed \
                --backend='nccl' \
                --device='cuda' \
                --dtype='float32' \
                --compile=False
        done
    done
done
