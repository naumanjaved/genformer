#!/bin/bash -l

python3 train_model_atac_rna.py \
            --tpu_name="node-1" \
            --tpu_zone="us-central1-a" \
            --wandb_project="paired_rna_atac" \
            --wandb_user="njaved" \
            --wandb_sweep_name="paired_rna_atac" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer/rampage_finetune/524k/paired_atac_rampage" \
            --gcs_path_holdout="gs://genformer/rampage_finetune/524k/paired_atac_rampage_holdout" \
            --input_length=524288 \
            --output_length=4096 \
            --output_length_ATAC=131072 \
            --final_output_length=4092 \
            --max_shift=4 \
            --batch_size=4 \
            --num_epochs=60 \
            --train_examples=40000 \
            --val_examples=19917  \
            --BN_momentum=0.90 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://genformer/rampage_finetune/models" \
            --model_save_basename="paired_rna_atac" \
            --lr_base1="1.0e-04" \
            --lr_base2="2.0e-04" \
            --decay_frac="0.005" \
            --gradient_clip="1.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="7" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="4" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=50 \
            --load_init_FT="False" \
            --load_init="False" \
            --rectify="True" \
            --filter_list_seq="512,512,768,768,1024,1024" \
            --filter_list_atac="32,64" \
            --atac_scale="0.05" \
            --atac_mask_dropout=0.05 \
            --atac_mask_dropout_val=0.05 \
            --random_mask_size="512" \
            --log_atac="False" \
            --final_point_scale="6" \
            --seed=25 \
            --val_data_seed=25 \
            --atac_corrupt_rate="20" \
            --use_motif_activity="True" \
            --use_atac="True" \
            --use_seq="True" \
            --loss_type="poisson" \
            --total_weight_loss="0.20" \
            --seq_mask="True"
