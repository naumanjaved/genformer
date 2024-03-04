#!/bin/bash -l

python3 train_model.py \
            --tpu_name="node-1" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_rampage_ft" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_rampage_ft_test" \
            --gcs_path_TSS="gs://enformer_baseline/human_tss/tfrecords" \
            --epsilon=1.0e-8 \
            --num_parallel=4 \
            --savefreq=1 \
            --test_examples=2032 \
            --num_targets=50 \
            --checkpoint_path="gs://enformer_baseline/models/enformer_baseline_2024-03-01_15:19:09_ENFORMER_LR1-1e-06_LR2-0.0001_GC-0.2_init-True_enformer_baseline_2024-03-01_15:19:00-28.data-00000-of-00001"