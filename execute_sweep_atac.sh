#!/bin/bash -l

python3 train_model_atac.py \
            --tpu_name="pod1" \
            --tpu_zone="us-central1-a" \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_europe_west_copy/atac_pretrain/524k/genformer_atac_pretrain_globalacc_conv_fpm" \
            --gcs_path_holdout="gs://genformer_europe_west_copy/atac_pretrain/524k/genformer_atac_pretrain_globalacc_conv_fpm_valid" \
            --input_length=524288 \
            --output_length=4096 \
            --output_length_ATAC=131072 \
            --final_output_length=4092 \
            --max_shift=4 \
            --batch_size=4 \
            --num_epochs=30 \
            --train_examples=1000000 \
            --val_examples_ho=38880 \
            --BN_momentum=0.90 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.0000005 \
            --model_save_dir="gs://genformer_europe_west_copy/atac_pretrain/models" \
            --model_save_basename="genformer" \
            --lr_base="1.0e-04" \
            --decay_frac="0.10" \
            --gradient_clip="1.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="8" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="4" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=1 \
            --load_init="False" \
            --rectify="True" \
            --filter_list_seq="512,512,768,768,1024,1024" \
            --filter_list_atac="32,64" \
            --atac_mask_dropout=0.15 \
            --atac_mask_dropout_val=0.15 \
            --log_atac="False" \
            --random_mask_size="1536" \
            --use_atac="True" \
            --final_point_scale="6" \
            --use_seq="True" \
            --seed=25 \
            --val_data_seed=25 \
            --atac_corrupt_rate="15" \
            --use_motif_activity="True" \
            --num_epochs_to_start="0" \
            --total_weight_loss="0.15" \
            --use_rot_emb="True" \
            --best_val_loss=100.0 \
            --loss_type="poisson"
