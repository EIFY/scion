for mode in 'Sign' 'ColNorm' 'RowNorm' 
do
    for width in 256 512 1024 2048
    do
        for lr in 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.00195312 0.0009765625 0.00048828125 0.000244140625
        do
            for seed in 1 2 3
            do
                head_size=64
                n_heads=$((width / head_size))
                out_dir="mup_examples/mutransfer_lr_shakespeare_char/scion_full/out/mode${mode}_width${width}_depth2_seed${seed}_lr${lr}"
                python train.py \
                    --out_dir=$out_dir \
                    --eval_on_end=True \
                    --eval_iters=200 \
                    --skip_val_loss=False \
                    --eval_only=False \
                    --log_interval=1 \
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
                    --scion_mode=$mode \
                    --scion_unconstrained=True \
                    --seed=$seed \
                    --backend='nccl' \
                    --device='cuda' \
                    --dtype='float32' \
                    --compile=False
            done
        done
    done
done
