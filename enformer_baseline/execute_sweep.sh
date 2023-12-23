#!/bin/bash -l

python3 train_model.py \
            --tpu_name="node-6" \
            --tpu_zone="us-central1-a" \
            --wandb_project="paired_rampage_atac" \
            --wandb_user="njaved" \
            --wandb_sweep_name="paired_rampage_atac" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/paired_rampage_atac/enformer_baseline/tfrecords_ends_only"\
            --gcs_path_TSS="gs://picard-testing-176520/paired_rampage_atac/enformer_baseline/tfrecords_tss_ends_only" \
            --num_epochs=50 \
            --warmup_frac=0.146 \
            --patience=30\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/paired_rampage_atac/enformer_baseline/models" \
            --model_save_basename="enformer_baseline" \
            --lr_base1="5.0e-06" \
            --lr_base2="5.0e-04" \
            --gradient_clip="0.2" \
            --epsilon=1.0e-8 \
            --num_parallel=4 \
            --savefreq=25 \
            --train_examples=70 \
            --val_examples=2213 \
            --val_examples_TSS=2135 \
            --num_targets=54 \
            --use_enformer_weights="True" \
            --enformer_checkpoint_path="sonnet_weights" \
            --freeze_trunk="True"
