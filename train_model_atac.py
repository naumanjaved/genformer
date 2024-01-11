import time
import os
import argparse
import wandb
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Environment configuration for TensorFlow
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'

## custom modules
import src.models.aformer_atac as genformer
import src.optimizers as optimizers
import training_utils_atac as training_utils

# Function to parse boolean string values
def parse_bool_str(input_str):
    if input_str in ['False', 'false', 'FALSE', 'F']:
        return False
    return True

# Main function definition
def main():
    # set up argument parser
    parser = argparse.ArgumentParser(
        description='process input for genformer training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()

    # defining sweep options, parameters are specified by execute_sweep.sh
    # -----------------------------------------------------------------------------------------------------------
    sweep_config = {
            "name" : args.wandb_sweep_name,
            'method': "grid",
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'input_length': {'values': [args.input_length]},
                'output_length': {'values': [args.output_length]},
                'output_length_ATAC': {'values': [args.output_length_ATAC]},
                'final_output_length': {'values': [args.final_output_length]},
                'output_res': {'values': [args.output_res]},
                'dropout_rate': {'values': [float(x) for x in args.dropout_rate.split(',')]},
                'pointwise_dropout_rate': {'values': [float(x) for x in args.pointwise_dropout_rate.split(',')]},
                'lr_base': {'values':[float(x) for x in args.lr_base.split(',')]},
                'gradient_clip': {'values': [float(x) for x in args.gradient_clip.split(',')]},
                'decay_frac': {'values': [float(x) for x in args.decay_frac.split(',')]},
                'num_transformer_layers': {'values': [int(x) for x in args.num_transformer_layers.split(',')]},
                'num_heads': {'values': [int(x) for x in args.num_heads.split(',')]},
                'num_random_features': {'values':[int(x) for x in args.num_random_features.split(',')]},
                'kernel_transformation': {'values':[args.kernel_transformation]},
                'epsilon': {'values':[args.epsilon]},
                'load_init': {'values':[parse_bool_str(x) for x in args.load_init.split(',')]},
                'filter_list_seq': {'values': [[int(x) for x in args.filter_list_seq.split(',')]]},
                'filter_list_atac': {'values': [[int(x) for x in args.filter_list_atac.split(',')]]},
                'BN_momentum': {'values': [args.BN_momentum]},
                'atac_mask_dropout': {'values': [args.atac_mask_dropout]},
                'atac_mask_dropout_val': {'values': [args.atac_mask_dropout_val]},
                'rectify': {'values':[parse_bool_str(x) for x in args.rectify.split(',')]},
                'log_atac': {'values':[parse_bool_str(x) for x in args.log_atac.split(',')]},
                'use_atac': {'values':[parse_bool_str(x) for x in args.use_atac.split(',')]},
                'use_seq': {'values':[parse_bool_str(x) for x in args.use_seq.split(',')]},
                'random_mask_size': {'values':[int(x) for x in args.random_mask_size.split(',')]},
                'final_point_scale': {'values':[int(x) for x in args.final_point_scale.split(',')]},
                'seed': {'values':[args.seed]},
                'val_data_seed': {'values':[args.val_data_seed]},
                'atac_corrupt_rate': {'values': [int(x) for x in args.atac_corrupt_rate.split(',')]},
                'use_motif_activity': {'values': [parse_bool_str(x) for x in args.use_motif_activity.split(',')]},
                'num_epochs_to_start': {'values': [int(x) for x in args.num_epochs_to_start.split(',')]},
                'loss_type': {'values': [str(x) for x in args.loss_type.split(',')]},
                'total_weight_loss': {'values': [float(x) for x in args.total_weight_loss.split(',')]},
                'use_rot_emb': {'values':[parse_bool_str(x) for x in args.use_rot_emb.split(',')]},
                'best_val_loss': {'values':[float(args.best_val_loss)]},
                'checkpoint_path': {'values':[args.checkpoint_path]}
                }
    }
    '''
    now that wandb optional parameters are specified, 
    track the remaining arguments and specify main training loop logic
    '''

    def sweep_train(config_defaults=None):
        strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone) # initialize TPU
        mixed_precision.set_global_policy('mixed_bfloat16')
        g = tf.random.Generator.from_seed(args.seed) # training data random seed init
        g_val = tf.random.Generator.from_seed(args.val_data_seed) # validation data random seed init

        with strategy.scope(): ## keep remainder of parameter initialization within TPU/GPU strategy scope
            config_defaults = {"lr_base": 0.01 }### will be overwritten by sweep config
            wandb.init(config=config_defaults,
                       project= args.wandb_project,
                       entity=args.wandb_user)
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.gcs_path_holdout=args.gcs_path_holdout
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_examples=args.train_examples
            wandb.config.val_examples_ho=args.val_examples_ho
            wandb.config.batch_size=args.batch_size
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift

            wandb.config.crop_size = (wandb.config.output_length - wandb.config.final_output_length) // 2
            run_name = '_'.join([str(int(wandb.config.input_length) / 1000)[:4].rstrip('.') + 'k',
                                 'LR-' + str(wandb.config.lr_base),
                                 'C-' + '_'.join([str(x) for x in wandb.config.filter_list_seq]),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'motif-' + str(wandb.config.use_motif_activity)])
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            run_name = run_name + "_" + date_string
            wandb.run.name = run_name 

            # TFrecord dataset options
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False
            options_val = tf.data.Options()
            options_val.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options_val.deterministic=False
            mixed_precision.set_global_policy('mixed_bfloat16')


            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS # num total examples per step across all replicas
            print('global batch size:', GLOBAL_BATCH_SIZE)

            wandb.config.update({"train_steps": wandb.config.train_examples // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps_ho" : wandb.config.val_examples_ho // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": wandb.config.train_examples // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)

            # create the dataset iterators, one for training, one for holdout validation
            train_human, data_val_ho = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path, wandb.config.gcs_path_holdout,
                                                                GLOBAL_BATCH_SIZE, wandb.config.input_length,
                                                                wandb.config.max_shift, wandb.config.output_length_ATAC,
                                                                wandb.config.output_length, wandb.config.crop_size,
                                                                wandb.config.output_res, args.num_parallel, args.num_epochs,
                                                                strategy, options,options_val, wandb.config.atac_mask_dropout,
                                                                wandb.config.atac_mask_dropout_val,
                                                                wandb.config.random_mask_size, wandb.config.log_atac,
                                                                wandb.config.use_atac, wandb.config.use_seq, wandb.config.seed,
                                                                wandb.config.val_data_seed, wandb.config.atac_corrupt_rate,
                                                                wandb.config.val_steps_ho, wandb.config.use_motif_activity,
                                                                g, g_val)

            print('created dataset iterators')
            print(wandb.config)

            # initialize model
            model = genformer.genformer(kernel_transformation=wandb.config.kernel_transformation,
                                    dropout_rate=wandb.config.dropout_rate,
                                    pointwise_dropout_rate=wandb.config.pointwise_dropout_rate,
                                    input_length=wandb.config.input_length,
                                    output_length=wandb.config.output_length,
                                    final_output_length=wandb.config.final_output_length,
                                    num_heads=wandb.config.num_heads,
                                    numerical_stabilizer=0.0000001,
                                    nb_random_features=wandb.config.num_random_features,
                                    max_seq_length=wandb.config.output_length,
                                    norm=True,
                                    BN_momentum=wandb.config.BN_momentum,
                                    normalize = True,
                                    seed = wandb.config.seed,
                                    num_transformer_layers=wandb.config.num_transformer_layers,
                                    final_point_scale=wandb.config.final_point_scale,
                                    filter_list_seq=wandb.config.filter_list_seq,
                                    filter_list_atac=wandb.config.filter_list_atac,
                                    use_rot_emb=wandb.config.use_rot_emb)

            print('initialized model')

            # initialize optimizer with warmup and cosine decay
            scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
            scheduler=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base,
                                         warmup_steps=10000,
                                         decay_schedule_fn=scheduler)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=scheduler, 
                                                    epsilon=wandb.config.epsilon,
                                                    weight_decay=1.0e-05,
                                                    global_clipnorm=wandb.config.gradient_clip)
            optimizer.exclude_from_weight_decay(var_names = ['bias', 'batch_norm','layer_norm', 
                                                            'BN', 'LN', 'LayerNorm','BatchNorm'])
            metric_dict = {} # initialize dictionary to store metrics

            ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer=optimizer, 
                                                            model=model)
            checkpoint_dir = wandb.config.model_save_dir + \
                                '/' + wandb.config.model_save_basename + \
                                    "_" + wandb.run.name
            
            manager = tf.train.CheckpointManager(ckpt, 
                                                 directory=checkpoint_dir,
                                                 max_to_keep=10)

            # initialize functions for training and validation steps
            train_step, val_step, build_step, metric_dict = \
                training_utils.return_train_val_functions(
                    model=model,
                    optimizer=optimizer,
                    strategy=strategy,
                    metric_dict=metric_dict,
                    num_replicas=NUM_REPLICAS,
                    loss_type=wandb.config.loss_type,
                    total_weight=wandb.config.total_weight_loss
                )

            val_losses = []
            if wandb.config.load_init: # if loading pretrained model, initialize the best val loss from previous run
                val_losses.append(wandb.config.best_val_loss)
            patience_counter = 0 # simple patience counter for early stopping
            stop_criteria = False
            best_epoch = 0 # track epoch with best validation loss 

            for epoch_i in range(1, wandb.config.num_epochs+1):
                step_num = (wandb.config.num_epochs_to_start+epoch_i) * \
                            wandb.config.train_steps * GLOBAL_BATCH_SIZE
                if epoch_i == 1: # if first epoch, build model which allows for weight loading
                    if wandb.config.load_init:
                        status = ckpt.restore(tf.train.latest_checkpoint(wandb.config.checkpoint_path))
                        status.assert_existing_objects_matched()
                        print(optimizer.lr.values[0])
                        print(optimizer.iterations.values[0])
                        print('restored from checkpoint')
                        print('restore iterator to state from last checkpoint...')
                        skip_steps=wandb.config.train_steps * wandb.config.num_epochs_to_start
                        print('skipping ' + str(skip_steps) + ' steps...')
                        @tf.function
                        def iterate():
                            for skip_step in range(skip_steps):
                                next(train_human)
                        iterate()

                # main training step 
                print('starting epoch_', str(epoch_i))
                start = time.time()
                for step in range(wandb.config.train_steps):
                    strategy.run(train_step, args=(next(train_human),))

                train_loss = metric_dict['train_loss'].result().numpy() * NUM_REPLICAS # multiply by NUM_REPLICAS to get total loss
                print('train_loss: ' + str(train_loss))

                wandb.log({'train_loss': train_loss},
                          step=step_num)
                wandb.log({'ATAC_pearsons_tr': metric_dict['ATAC_PearsonR_tr'].result()['PearsonR'].numpy(),
                           'ATAC_R2_tr': metric_dict['ATAC_R2_tr'].result()['R2'].numpy()},
                          step=step_num)
                duration = (time.time() - start) / 60.

                print('completed epoch ' + str(epoch_i) + ' - duration(mins): ' + str(duration))

                # main validation step:
                # - run the validation loop
                # - return the true and predicted values to allow for plotting and other metrics
                start = time.time()

                for k in range(wandb.config.val_steps_ho):
                    strategy.run(val_step, args=(next(data_val_ho),))

                val_loss = NUM_REPLICAS * metric_dict['val_loss'].result().numpy() # multiply by NUM_REPLICAS to get total loss 
                print('val_loss: ' + str(val_loss))

                val_losses.append(val_loss)
                wandb.log({'val_loss': val_loss}, step=step_num)
                print('ATAC_pearsons: ' + str(metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()))
                print('ATAC_R2: ' + str(metric_dict['ATAC_R2'].result()['R2'].numpy()))
                wandb.log({'ATAC_pearsons': metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy(),
                           'ATAC_R2': metric_dict['ATAC_R2'].result()['R2'].numpy()},
                          step=step_num)

                duration = (time.time() - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation - duration(mins): ' + str(duration))

                # Start early stopping checks:
                # - After epoch one if not loading from a checkpoint
                # - Immediately (epoch 0) if loading from a checkpoint, using the provided best loss
                if (epoch_i > 0 and not wandb.config.load_init) or wandb.config.load_init: 
                    stop_criteria, patience_counter, best_epoch = \
                        training_utils.early_stopping(
                            current_val_loss=val_losses[-1],             # Last value from val_losses
                            logged_val_losses=val_losses,                # Full list of val_losses
                            current_epoch=epoch_i,                       # Current epoch number
                            best_epoch=best_epoch,                       # Best epoch so far
                            save_freq=args.savefreq,                     # Frequency of saving the model
                            patience=wandb.config.patience,              # Patience for early stopping
                            patience_counter=patience_counter,           # Current patience counter
                            min_delta=wandb.config.min_delta,            # Minimum change for early stopping
                        )

                if (epoch_i % args.savefreq) == 0:
                    ckpt.step.assign_add(step_num)
                    save_path = manager.save()
                    print('saving model at: epoch ' + str(epoch_i))

                print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items(): # reset metrics for new epoch
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break

            print('best model was at: epoch ' + str(best_epoch))


    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
