# genformer repository

**genformer** learns representations of accessible sequence via "masked accessibility modeling" which can be used for downstream fine-tuning tasks

## dataset inputs
See https://app.terra.bio/#workspaces/epigenomics/gro_nn for data processing workflows
and input data. 

## main files
For pre-training(masked atac prediction, _atac suffix files):
 * execute_sweep_atac.sh - training bash script where you can define hyperparameters
 * training_utils_atac.py - define train and validation steps, data loading and augmentation, masking, early stopping, model saving
 * train_model_atac.py - define main training loop, argument parsing, wandb initialization code, TPU initialization code
 * src/models/aformer_atac.py - main model file
 * src/layers/layers.py - all custom layers
 * src/layers/fast_attention_rpe_genformer1.py - linear attention code with rotary positional encodings

Files for fine-tuning for RAMPAGE prediction follow a similar structure

## training

Define hyper- and sweep parameters in execute_sweep_atac.sh


