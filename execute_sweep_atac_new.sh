#!/bin/bash -l

export TPU_LOAD_LIBRARY=0
export TPU_NAME=$1
export ZONE=$2

python3 train_model_atac.py \
            --tpu_name=$1 \
            --tpu_zone=$2 \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://$3/atac_pretrain/524k/genformer_atac_pretrain_globalacc_conv_fpm" \
            --gcs_path_holdout="gs://$3/atac_pretrain/524k/genformer_atac_pretrain_globalacc_conv_fpm_valid" \
            --input_length=524288 \
            --output_length=4096 \
            --output_length_ATAC=131072 \
            --final_output_length=4092 \
            --max_shift=4 \
            --batch_size=1 \
            --val_examples_ho=38880 \
            --BN_momentum=0.90 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.0000005 \
            --model_save_dir="gs://$3/atac_pretrain/models" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="10" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=1 \
            --rectify="True" \
            --filter_list_seq="512,640,640,768,896,1024" \
            --filter_list_atac="32,64" \
            --atac_mask_dropout=0.15 \
            --atac_mask_dropout_val=0.15 \
            --log_atac="False" \
            --random_mask_size="1536" \
            --use_atac="True" \
            --final_point_scale="6" \
            --use_seq="True" \
            --atac_corrupt_rate="15" \
            --use_motif_activity="True" \
            --total_weight_loss="0.15" \
            --use_rot_emb="True" \
            --lr_base="1.0e-04" \
            --decay_frac="0.10" \
            --gradient_clip="1.0" \
            --seed=1 \
            --val_data_seed=19 \
            --loss_type="poisson" \
            --model_save_basename="genformer" \
            --warmup_steps=5000 \
            --decay_steps=500000 \
            --weight_decay=1.0e-05
