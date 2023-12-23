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
import src.models.aformer_atac_rna as genformer
import src.optimizers as optimizers
import training_utils_atac_rna as training_utils
import src.load_weights_atac_rna as load_weights_atac_rna

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
                'lr_base1': {'values':[float(x) for x in args.lr_base1.split(',')]},
                'lr_base2': {'values':[float(x) for x in args.lr_base2.split(',')]},
                'atac_scale': {'values':[float(x) for x in args.atac_scale.split(',')]},
                'gradient_clip': {'values': [float(x) for x in args.gradient_clip.split(',')]},
                'decay_frac': {'values': [float(x) for x in args.decay_frac.split(',')]},
                'num_transformer_layers': {'values': [int(x) for x in args.num_transformer_layers.split(',')]},
                'num_heads': {'values': [int(x) for x in args.num_heads.split(',')]},
                'num_random_features': {'values':[int(x) for x in args.num_random_features.split(',')]},
                'kernel_transformation': {'values':[args.kernel_transformation]},
                'epsilon': {'values':[args.epsilon]},
                'load_init': {'values':[parse_bool_str(x) for x in args.load_init.split(',')]},
                'load_init_FT': {'values':[parse_bool_str(x) for x in args.load_init_FT.split(',')]},
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
                'seq_mask': {'values':[parse_bool_str(x) for x in args.seq_mask.split(',')]},
                'best_val_loss': {'values':[float(args.best_val_loss)]}
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
                                 'load-' + str(wandb.config.load_init),
                                 'LR-' + str(wandb.config.lr_base),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'TF-' + str(wandb.config.use_motif_activity)])
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            wandb.run.name = run_name + "_" + date_string

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
                                                                g, g_val, wandb.config.seq_mask)

            print('created dataset iterators')
            print(wandb.config)

            inits=None
            if wandb.config.load_init_FT:
                print('loading fine-tuning weights')
                inits=load_weights_atac_rna.get_initializers_genformer_ft(wandb.config.checkpoint_path,
                                                                          wandb.config.num_transformer_layers)

            print('initializing model')
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
                                    load_init=wandb.config.load_init_FT,
                                    inits=inits,
                                    seed = wandb.config.seed,
                                    num_transformer_layers=wandb.config.num_transformer_layers,
                                    final_point_scale=wandb.config.final_point_scale,
                                    filter_list_seq=wandb.config.filter_list_seq,
                                    filter_list_atac=wandb.config.filter_list_atac,
                                    use_rot_emb=wandb.config.use_rot_emb)

            print('initialized model')

            # initialize optimizer with warmup and cosine decay
            scheduler1= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base1,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
            scheduler1=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1,
                                         warmup_steps=10000,
                                         decay_schedule_fn=scheduler1)
            optimizer1 = tf.keras.optimizers.AdamW(learning_rate=scheduler1, 
                                                    epsilon=wandb.config.epsilon,
                                                    weight_decay=1.0e-03,
                                                    global_clipnorm=wandb.config.gradient_clip)
            optimizer1.exclude_from_weight_decay(var_names = ['bias', 'batch_norm','layer_norm', 
                                                            'BN', 'LN', 'LayerNorm','BatchNorm'])
            scheduler2= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base2,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
            scheduler2=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2,
                                         warmup_steps=10000,
                                         decay_schedule_fn=scheduler2)
            optimizer2 = tf.keras.optimizers.AdamW(learning_rate=scheduler2, 
                                                    epsilon=wandb.config.epsilon,
                                                    weight_decay=1.0e-03,
                                                    global_clipnorm=wandb.config.gradient_clip)
            optimizer2.exclude_from_weight_decay(var_names = ['bias', 'batch_norm','layer_norm', 
                                                            'BN', 'LN', 'LayerNorm','BatchNorm'])
            optimizers_in = optimizer1,optimizer2
            metric_dict = {} # initialize dictionary to store metrics

            # initialize functions for training and validation steps
            train_step, val_step, build_step, metric_dict = \
                training_utils.return_train_val_functions(
                    model=model,
                    optimizers_in=optimizers_in,
                    strategy=strategy,
                    metric_dict=metric_dict,
                    num_replicas=NUM_REPLICAS,
                    atac_scale=wandb.config.atac_scale,
                    loss_type=wandb.config.loss_type,
                    total_weight=wandb.config.total_weight
                )


            global_step = 0
            val_losses = []
            if wandb.config.load_init: # if loading pretrained model, initialize the best val loss from previous run
                val_losses.append(wandb.config.best_val_loss)
            val_pearsons = [] # track pearsons per epoch
            val_R2 = [] # track r2 per epoch
            patience_counter = 0 # simple patience counter for early stopping
            stop_criteria = False
            best_epoch = 0 # track epoch with best validation loss 

            for epoch_i in range(1, wandb.config.num_epochs+1):
                step_num = (wandb.config.num_epochs_to_start+epoch_i) * \
                            wandb.config.train_steps * GLOBAL_BATCH_SIZE
                if epoch_i == 1: # if first epoch, build model which allows for weight loading
                    print('building model')
                    build_step(data_val_ho)
                    if wandb.config.load_init:
                        model.load_weights(args.checkpoint_path + "/saved_model")
                        print('built and loaded model')
                    total_params = 0
                    for k in model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params))

                # main training step 
                print('starting epoch_', str(epoch_i))
                start = time.time()
                for step in range(wandb.config.train_steps):
                    strategy.run(train_step, args=(next(train_human),))

                train_loss = metric_dict['train_loss'].result().numpy() * NUM_REPLICAS # multiply by NUM_REPLICAS to get total loss
                train_loss_rna = metric_dict['train_loss_rna'].result().numpy() * NUM_REPLICAS # multiply by NUM_REPLICAS to get total loss
                train_loss_atac = metric_dict['train_loss_atac'].result().numpy() * NUM_REPLICAS # multiply by NUM_REPLICAS to get total loss
                print('train_loss: ' + str(train_loss))
                print('train_loss_rna: ' + str(train_loss_rna))
                print('train_loss_atac: ' + str(train_loss_atac))

                wandb.log({'train_loss': train_loss,
                            'train_loss_atac': train_loss_atac,
                            'train_loss_rna': train_loss_rna}, step=step_num)
                wandb.log({'ATAC_pearsons_tr': metric_dict['ATAC_PearsonR_tr'].result()['PearsonR'].numpy(),
                           'ATAC_R2_tr': metric_dict['ATAC_R2_tr'].result()['R2'].numpy()},
                          step=step_num)
                wandb.log({'RNA_pearsons_tr': metric_dict['RNA_PearsonR_tr'].result()['PearsonR'].numpy(),
                           'RNA_R2_tr': metric_dict['RNA_R2_tr'].result()['R2'].numpy()},
                          step=step_num)
                duration = (time.time() - start) / 60.

                print('completed epoch ' + str(epoch_i) + ' - duration(mins): ' + str(duration))

                # main validation step:
                # - run the validation loop
                # - return the true and predicted values to allow for plotting and other metrics
                start = time.time()
                pred_list = [] # list to store predictions
                true_list = [] # list to store true values
                cell_type_list = [] # list to store predictions
                gene_list = [] # list to store true values
                for k in range(wandb.config.val_steps_ho):
                    true, pred,gene,cell_type = strategy.run(val_step, args=(next(data_val_ho),))
                    for x in strategy.experimental_local_results(true): # flatten the true values
                        true_list.append(tf.reshape(x, [-1]))
                    for x in strategy.experimental_local_results(pred): # flatten the pred values
                        pred_list.append(tf.reshape(x, [-1]))
                    for x in strategy.experimental_local_results(cell_type): # flatten the true values
                        cell_type_list.append(tf.reshape(x, [-1]))
                    for x in strategy.experimental_local_results(gene): # flatten the pred values
                        gene_list.append(tf.reshape(x, [-1]))

                figures,overall_corr = training_utils.make_plots(tf.concat(true_list,0),
                                                                 tf.concat(pred_list,0),
                                                                 tf.concat(cell_type_list,0),
                                                                 tf.concat(gene_list,0), 5000)

                fig_cell_spec, fig_gene_spec, fig_overall=figures 

                cell_specific_corrs, gene_specific_corrs, \
                    cell_specific_corrs_raw, gene_specific_corrs_raw= overall_corr

                val_loss = NUM_REPLICAS * metric_dict['val_loss'].result().numpy() # multiply by NUM_REPLICAS to get total loss 
                val_loss_rna = NUM_REPLICAS * metric_dict['val_loss_rna'].result().numpy() # multiply by NUM_REPLICAS to get total loss 
                val_loss_atac = NUM_REPLICAS * metric_dict['val_loss_atac'].result().numpy() # multiply by NUM_REPLICAS to get total loss 
                print('val_loss: ' + str(val_loss))
                print('val_loss_rna: ' + str(val_loss_rna))
                print('val_loss_atac: ' + str(val_loss_atac))

                val_losses.append(val_loss)
                wandb.log({'val_loss': val_loss,
                            'val_loss_rna': val_loss_rna,
                            'val_loss_atac': val_loss_atac},
                           step=step_num)
                val_pearsons.append(metric_dict['RNA_PearsonR'].result()['PearsonR'].numpy())
                print('ATAC_pearsons: ' + str(metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()))
                print('ATAC_R2: ' + str(metric_dict['ATAC_R2'].result()['R2'].numpy()))
                print('RNA_pearsons: ' + str(metric_dict['RNA_PearsonR'].result()['PearsonR'].numpy()))
                print('RNA_R2: ' + str(metric_dict['RNA_R2'].result()['R2'].numpy()))
                print('cell_specific_correlation: ' + str(cell_specific_corrs))
                print('gene_specific_correlation: ' + str(gene_specific_corrs))
                wandb.log({'ATAC_pearsons': metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy(),
                           'ATAC_R2': metric_dict['ATAC_R2'].result()['R2'].numpy(),
                           'RNA_pearsons': metric_dict['RNA_PearsonR'].result()['PearsonR'].numpy(),
                           'RNA_R2': metric_dict['RNA_R2'].result()['R2'].numpy()},
                          step=step_num)

                wandb.log({'gene_spec_mean_corrs': gene_specific_corrs,
                            'cell_spec_mean_corrs': cell_specific_corrs,
                            'gene_spec_mean_corrs_raw': gene_specific_corrs_raw,
                            'cell_spec_mean_corrs_raw': cell_specific_corrs_raw},
                            step=epoch_i)
                wandb.log({'hg_OVERALL_TSS_predictions': fig_overall,
                            'cross_cell_dist': fig_cell_spec,
                            'cross_gene_dist': fig_gene_spec},
                            step=epoch_i)

                wandb.log({'overall_predictions': figures},
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
                            current_pearsons=val_pearsons[-1],           # Last value from val_pearsons
                            logged_pearsons=val_pearsons,                # Full list of val_pearsons
                            current_epoch=epoch_i,                       # Current epoch number
                            best_epoch=best_epoch,                       # Best epoch so far
                            save_freq=args.savefreq,                     # Frequency of saving the model
                            patience=wandb.config.patience,              # Patience for early stopping
                            patience_counter=patience_counter,           # Current patience counter
                            min_delta=wandb.config.min_delta,            # Minimum change for early stopping
                            model=model,                                 # Model to be used
                            save_directory=wandb.config.model_save_dir,  # Directory for saving the model
                            saved_model_basename=wandb.config.model_save_basename + "_" + wandb.run.name
                        )

                print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items(): # reset metrics for new epoch
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break

            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            model_save_path = os.path.join(
                wandb.config.model_save_dir,
                wandb.config.model_save_basename + "_" + wandb.run.name,
                "final",
                "saved_model"
            )
            model.save_weights(model_save_path)


    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()