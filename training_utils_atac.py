import os
import multiprocessing
import random
import numpy as np

os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
from tensorflow import strings as tfs
import src.metrics as metrics 
import src.utils
from src.losses import poisson_multinomial


tf.keras.backend.set_floatx('float32')

def return_train_val_functions(model, optimizer,
                               strategy, metric_dict, num_replicas,
                               loss_type,total_weight=0.15):
    """Return training, validation, and build step functions
    Args:
        model: the input genformer model
        optimizer: input optimizer, e.g. Adam
        strategy: input distribution strategy
        metric_dict: dictionary of metrics to track
        num_replicas: num replicas for distributed training
        gradient_clip: gradient clipping value
        loss_type: poisson or poisson_multinomial
        total_weight: total weight for the poisson_multinomial loss
    """

    # initialize metrics
    metric_dict["train_loss"] = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    metric_dict["val_loss"] = tf.keras.metrics.Mean("val_loss",  dtype=tf.float32)

    metric_dict["train_loss_um"] = tf.keras.metrics.Mean("train_loss_um", dtype=tf.float32)
    metric_dict["val_loss_um"] = tf.keras.metrics.Mean("val_loss_um",  dtype=tf.float32)

    metric_dict['ATAC_PearsonR_tr'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2_tr'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    metric_dict['ATAC_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    metric_dict['ATAC_PearsonR_um'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2_um'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    if loss_type == 'poisson_multinomial':
        def loss_fn(y_true,y_pred, total_weight=total_weight, epsilon=1e-6, rescale=True):
            return poisson_multinomial(y_true, y_pred, total_weight,epsilon,rescale=True)
    elif loss_type == 'poisson':
        loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    else:
        raise ValueError('loss_type not implemented')


    @tf.function(reduce_retracing=True)
    def dist_train_step(inputs): # main train step
        print('tracing training step!') # just to make sure no retracing occurring
        sequence,atac,mask,target,motif_activity =inputs

        input_tuple = sequence, atac, motif_activity

        with tf.GradientTape() as tape:
            output_profile = model(input_tuple, training=True)
            output_profile = tf.cast(output_profile,dtype=tf.float32)

            mask_indices = tf.where(mask == 1) # extract indices of masked bins
            # subset target and predictions to masked bins
            target_atac = tf.expand_dims(tf.expand_dims(tf.gather_nd(target, mask_indices), axis=0), axis=2)
            output_atac = tf.expand_dims(tf.expand_dims(tf.gather_nd(output_profile, mask_indices), axis=0), axis=2)

            ## masked loss 
            loss_masked = tf.reduce_mean(loss_fn(target_atac, output_atac)) * (1.0/num_replicas)

            ## unmasked_loss
            mask_indices_um = tf.where(mask == 0) # extract indices of unmasked bins
            # subset target and predictions to masked bins
            target_atac_um = tf.expand_dims(tf.expand_dims(tf.gather_nd(target, mask_indices_um), axis=0), axis=2)
            output_atac_um = tf.expand_dims(tf.expand_dims(tf.gather_nd(output_profile, mask_indices_um), axis=0), axis=2)
            loss_unmasked = tf.reduce_mean(loss_fn(target_atac_um, output_atac_um)) * (1.0/num_replicas)
            
            loss = loss_masked + loss_unmasked

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,  model.trainable_variables))

        # update metrics for train step, compute pearsons just over the masked regions
        metric_dict["train_loss"].update_state(loss)
        metric_dict["train_loss_um"].update_state(loss_unmasked)
        metric_dict['ATAC_PearsonR_tr'].update_state(target_atac, output_atac)
        metric_dict['ATAC_R2_tr'].update_state(target_atac, output_atac)

    @tf.function(reduce_retracing=True)
    def dist_val_step(inputs):  # main validation step 
        print('tracing validation step!')
        sequence,atac,mask,target,motif_activity =inputs
        input_tuple = sequence,atac,motif_activity
        output_profile = model(input_tuple, training=False)
        output_profile = tf.cast(output_profile,dtype=tf.float32)

        mask_indices = tf.where(mask == 1) # extract indices of masked bins
        target_atac = tf.expand_dims(tf.expand_dims(tf.gather_nd(target, mask_indices), axis=0), axis=2)
        output_atac = tf.expand_dims(tf.expand_dims(tf.gather_nd(output_profile, mask_indices), axis=0), axis=2)
        loss_masked = tf.reduce_mean(loss_fn(target_atac, output_atac)) * (1.0/num_replicas)

        ## unmasked_loss
        mask_indices_um = tf.where(mask == 0) # extract indices of unasked bins
        target_atac_um = tf.expand_dims(tf.expand_dims(tf.gather_nd(target, mask_indices_um), axis=0), axis=2)
        output_atac_um = tf.expand_dims(tf.expand_dims(tf.gather_nd(output_profile, mask_indices_um), axis=0), axis=2)
        loss_unmasked = tf.reduce_mean(loss_fn(target_atac_um, output_atac_um)) * (1.0/num_replicas)
        
        loss = loss_masked + loss_unmasked

        metric_dict["val_loss"].update_state(loss)
        metric_dict["val_loss_um"].update_state(loss_unmasked)
        metric_dict['ATAC_PearsonR'].update_state(target_atac, output_atac)
        metric_dict['ATAC_R2'].update_state(target_atac, output_atac)
        metric_dict['ATAC_PearsonR_um'].update_state(target_atac_um, output_atac_um)
        metric_dict['ATAC_R2_um'].update_state(target_atac_um, output_atac_um)

    
    def build_step(iterator): # just to build the model
        @tf.function(reduce_retracing=True)
        def val_step(inputs):
            sequence,atac,mask,target,motif_activity =inputs
            input_tuple = sequence,atac,motif_activity
            model(input_tuple, training=False)

        strategy.run(val_step, args=(next(iterator),))

    return dist_train_step,dist_val_step,build_step,metric_dict

@tf.function
def deserialize_tr(serialized_example, g, use_motif_activity,
                   input_length = 524288, max_shift = 4, output_length_ATAC = 131072,
                   output_length = 4096, crop_size = 2, output_res = 128,
                   atac_mask_dropout = 0.15, mask_size = 1536, log_atac = False,
                   use_atac = True, use_seq = True, atac_corrupt_rate = 20):
    """
    Deserialize a serialized example from a TFRecordFile and apply various transformations
    and augmentations to the data. Among these, the input atac profile will have one atac peak
    masked, atac_mask_dropout percentage of the remaining profile will also be masked, and the
    sequence/targets will be randomly shifted and reverse complemented.

    Parameters:
    - serialized_example: Serialized example from a TFRecord file.
    - g: TensorFlow random number generator.
    - use_motif_activity: Whether to use TF activity  or provide random noise for ablations
    - input_length: expected length of input sequence
    - max_shift: maximum number of bases to shift the input sequence
    - output_length_ATAC: expected length of the ATAC profile
    - output_length: expected length of the target ATAC profile (should be output_length_ATAC // output_res)
    - crop_size: number of bins to crop from the edges of the target ATAC profile
    - output_res: resolution of the target ATAC profile, default is 128
    - atac_mask_dropout: percentage of the target ATAC profile to mask
    - mask_size: size of the masked region in the target ATAC profile, expressed as number of bp
    - log_atac: whether to log transform the input ATAC profile
    - use_atac: whether to use the input ATAC profile
    - use_seq: whether to use the input sequence
    - atac_corrupt_rate: rate at which to corrupt the input ATAC profile

    Returns:
    - Tuple of processed sequence, masked ATAC input, mask, ATAC output, and motif activity tensors.
    """


    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string), # sequence string
        'atac': tf.io.FixedLenFeature([], tf.string), # atac profile, input
        'peaks_center': tf.io.FixedLenFeature([], tf.string), # int 1D tensor showing peak center locations
        'motif_activity': tf.io.FixedLenFeature([], tf.string) #motif activity tensor, 693 x 1 
    }

    ## here we need to set up some random numbers to achieve data augmentation
    atac_mask_int = g.uniform([], 0, atac_corrupt_rate, dtype=tf.int32) # random increase ATAC masking w/ 1/atac_corrupt_rate prob. 
    randomish_seed = g.uniform([], 0, 100000000,dtype=tf.int32) # work-around to ensure random-ish stateless operations
    shift = g.uniform(shape=(), minval=0, maxval=max_shift, dtype=tf.int32)
    
    rev_comp = tf.math.round(g.uniform([], 0, 1)) #switch for random reverse complementation
    
    # parse out the actual data 
    data = tf.io.parse_example(serialized_example, feature_map)

    # sequence, get substring based on sequence shift, rev comp/mask/one hot 
    sequence = one_hot(tf.strings.substr(data['sequence'], shift,input_length))

    # atac input, cast to float32 
    atac = tf.cast(tf.ensure_shape(tf.io.parse_tensor(data['atac'], out_type=tf.float16), 
                                   [output_length_ATAC,1]),dtype=tf.float32)
    atac = atac + tf.math.abs(g.normal(atac.shape,mean=1.0e-04,stddev=1.0e-04,dtype=tf.float32))

    atac_target = atac ## store the target ATAC, as we will subsequently directly manipulate atac for masking
    # get peaks centers 
    peaks_center = tf.expand_dims(tf.io.parse_tensor(data['peaks_center'], out_type=tf.int32), 
                                  axis=1)
    peaks_c_crop = tf.slice(peaks_center, [crop_size,0], [output_length-2*crop_size,-1]) # crop at the outset 
    # TF activity, cast to float32 and expand dims to allow for processing by model input FC layers
    motif_activity = tf.ensure_shape(tf.io.parse_tensor(data['motif_activity'], out_type=tf.float16), [693])
    motif_activity = tf.cast(motif_activity,dtype=tf.float32)
    motif_activity = tf.expand_dims(motif_activity,axis=0)
    min_val = tf.reduce_min(motif_activity)
    max_val = tf.reduce_max(motif_activity)
    motif_activity = (motif_activity - min_val) / (max_val - min_val)
    motif_activity = motif_activity + \
        tf.math.abs(g.normal(motif_activity.shape,mean=0.0,stddev=0.0001,dtype=tf.float32))
    
    if not use_motif_activity: # if running ablation, just set TF activity to 0
        print('not using tf activity')
        motif_activity = tf.zeros_like(motif_activity)

    # generate ATAC mask 
    full_comb_mask, full_comb_mask_store = mask_ATAC_profile(
                                                output_length_ATAC,
                                                output_length,
                                                crop_size,
                                                mask_size,
                                                output_res,
                                                peaks_c_crop,
                                                randomish_seed,
                                                atac_mask_int,
                                                atac_mask_dropout)

    masked_atac = atac * full_comb_mask ## apply the mask to the input profile

    if log_atac:
        masked_atac = tf.math.log1p(masked_atac)

    ## input ATAC clipping
    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff

    #  reverse targets + peaks + mask if rev complementing the sequence
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        atac_target = tf.reverse(atac_target,axis=[0])
        masked_atac = tf.reverse(masked_atac,axis=[0])
        full_comb_mask = tf.reverse(full_comb_mask,axis=[0])
        full_comb_mask_store=tf.reverse(full_comb_mask_store,axis=[0])

    # generate the output atac profile by summing the inputs to a desired resolution
    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 2000.0 * tf.ones(atac_out.shape))) # soft clip the targets
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=2000.0) + diff
    atac_out = tf.slice(atac_out, [crop_size,0], [output_length-2*crop_size,-1]) # crop to desired length
    
    # in case we want to run ablation without these inputs
    if not use_atac:
        print('not using atac')
        masked_atac = tf.random.stateless_uniform(shape=[output_length_ATAC, 1], minval=0, maxval=150, 
                                                      seed=[randomish_seed+1,randomish_seed+3],
                                                      dtype=tf.float32)

    if not use_seq:
        print('not using sequence')
        random_sequence = tf.random.stateless_uniform(shape=[input_length], minval=0, maxval=4, 
                                                      seed=[randomish_seed+21,randomish_seed+2],
                                                      dtype=tf.int32)
        random_one_hot = tf.one_hot(random_sequence, depth=4)
        sequence = random_one_hot

    return tf.cast(tf.ensure_shape(sequence, 
                                   [input_length,4]),dtype=tf.bfloat16), \
                tf.cast(masked_atac,dtype=tf.bfloat16), \
                tf.cast(full_comb_mask_store,dtype=tf.int32), \
                tf.cast(atac_out,dtype=tf.float32), \
                tf.cast(motif_activity,dtype=tf.bfloat16)

@tf.function
def deserialize_val(serialized_example, g_val, use_motif_activity,
                   input_length = 524288, max_shift = 10, output_length_ATAC = 131072,
                   output_length = 4096, crop_size = 2, output_res = 128,
                   atac_mask_dropout = 0.15, mask_size = 1536, log_atac = False,
                   use_atac = True, use_seq = True):
    """
    Deserialize a serialized example from a TFRecordFile to return validation data. 
    For validation, we will mask one peak of the input atac profile  and apply atac_mask_dropout 
    percentage to remaining profile will also be masked. Also, the sequence/targets will be 
    randomly reverse complemented.

    Parameters:
    - serialized_example: Serialized example from a TFRecord file.
    - g: TensorFlow random number generator.
    - use_motif_activity: Whether to use TF activity  or provide random noise for ablations
    - input_length: expected length of input sequence
    - max_shift: maximum number of bases to shift the input sequence
    - output_length_ATAC: expected length of the ATAC profile
    - output_length: expected length of the target ATAC profile (should be output_length_ATAC // output_res)
    - crop_size: number of bins to crop from the edges of the target ATAC profile
    - output_res: resolution of the target ATAC profile, default is 128
    - atac_mask_dropout: percentage of the target ATAC profile to mask
    - mask_size: size of the masked region in the target ATAC profile, expressed as number of bp
    - log_atac: whether to log transform the input ATAC profile
    - use_atac: whether to use the input ATAC profile
    - use_seq: whether to use the input sequence
    Returns:
    - Tuple of processed sequence, masked ATAC, mask, ATAC output, and TF activity tensors.
    """

    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'motif_activity': tf.io.FixedLenFeature([], tf.string)
    }

    ## now parse out the actual data
    # parse out the actual data 
    data = tf.io.parse_example(serialized_example, feature_map)

    # atac input, cast to float32 
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'], out_type=tf.float16), [output_length_ATAC,1])
    atac = tf.cast(atac,dtype=tf.float32)
    atac = atac + tf.math.abs(g_val.normal(atac.shape,mean=1.0e-04,stddev=1.0e-04,dtype=tf.float32))

    atac_target = atac ## store the target ATAC, as we will subsequently directly manipulate atac for masking

    # set up a semi-random seem based on the number of
    # peaks and atac signal in the window
    # get peaks centers 
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'], out_type=tf.int32), [output_length])
    peaks_center = tf.expand_dims(peaks_center,axis=1)
    peaks_c_crop = tf.slice(peaks_center, [crop_size,0], [output_length-2*crop_size,-1]) # crop at the outset

    peaks_sum = tf.reduce_sum(peaks_center)

    randomish_seed = peaks_sum + tf.cast(tf.reduce_sum(atac),dtype=tf.int32)

    rev_comp = tf.math.round(g_val.uniform([], 0, 1)) #switch for random reverse complementation
    
    shift = g_val.uniform(shape=(), minval=0, maxval=max_shift, dtype=tf.int32)
    
    rev_comp = tf.math.round(g_val.uniform([], 0, 1)) #switch for random reverse complementation
    
    # sequence, get substring based on sequence shift, one_hot
    sequence = one_hot(tf.strings.substr(data['sequence'], shift,input_length))

    # motif activity, cast to float32 and expand dims to allow for processing by model input FC layers
    motif_activity = tf.ensure_shape(tf.io.parse_tensor(data['motif_activity'], out_type=tf.float16), [693])
    motif_activity = tf.cast(motif_activity,dtype=tf.float32)
    motif_activity = tf.expand_dims(motif_activity,axis=0)
    min_val = tf.reduce_min(motif_activity)
    max_val = tf.reduce_max(motif_activity)
    motif_activity = (motif_activity - min_val) / (max_val - min_val)
    motif_activity = motif_activity + \
        tf.math.abs(g_val.normal(motif_activity.shape,mean=0.0,stddev=0.0001,dtype=tf.float32))
    
    if not use_motif_activity: # if running ablation, just set TF activity to 0
        print('not using tf activity')
        motif_activity = tf.zeros_like(motif_activity)

    # generate ATAC mask 
    full_comb_mask, full_comb_mask_store = mask_ATAC_profile(
                                                output_length_ATAC,
                                                output_length,
                                                crop_size,
                                                mask_size,
                                                output_res,
                                                peaks_c_crop,
                                                randomish_seed,
                                                1, # set atac_mask_int to 1 to prevent increased masking used in training
                                                atac_mask_dropout)

    masked_atac = atac * full_comb_mask ## apply the mask to the input profile

    if log_atac:
        masked_atac = tf.math.log1p(masked_atac)

    ## input ATAC clipping
    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff

    #  reverse targets + peaks + mask if rev complementing the sequence
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        atac_target = tf.reverse(atac_target,axis=[0])
        masked_atac = tf.reverse(masked_atac,axis=[0])
        full_comb_mask_store=tf.reverse(full_comb_mask_store,axis=[0])

    # generate the output atac profile by summing the inputs to a desired resolution
    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 2000.0 * tf.ones(atac_out.shape))) # soft clip the targets
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=2000.0) + diff
    atac_out = tf.slice(atac_out, [crop_size,0], [output_length-2*crop_size,-1]) # crop to desired length

    # in case we want to run ablation without these inputs
    if not use_atac:
        print('not using atac')
        masked_atac = tf.random.stateless_uniform(shape=[output_length_ATAC, 1], minval=0, maxval=150, 
                                                      seed=[randomish_seed+1,randomish_seed+3],
                                                      dtype=tf.float32)

    if not use_seq:
        print('not using sequence')
        random_sequence = tf.random.stateless_uniform(shape=[input_length], minval=0, maxval=4, 
                                                      seed=[randomish_seed+21,randomish_seed+2],
                                                      dtype=tf.int32)
        random_one_hot = tf.one_hot(random_sequence, depth=4)
        sequence = random_one_hot

    return tf.cast(tf.ensure_shape(sequence, 
                                   [input_length,4]),dtype=tf.bfloat16), \
                tf.cast(masked_atac,dtype=tf.bfloat16), \
                tf.cast(full_comb_mask_store,dtype=tf.int32), \
                tf.cast(atac_out,dtype=tf.float32), \
                tf.cast(motif_activity,dtype=tf.bfloat16)

def return_dataset(gcs_path, split, batch, input_length, output_length_ATAC,
                   output_length, crop_size, output_res, max_shift, options,
                   num_parallel, num_epoch, atac_mask_dropout,
                   random_mask_size, log_atac, use_atac, use_seq, seed,
                   atac_corrupt_rate, validation_steps,
                   use_motif_activity, g):
    """
    return a tf dataset object for given gcs path
    """
    wc = "*.tfr"

    if split == 'train':
        list_files = tf.io.gfile.glob(os.path.join(gcs_path, split, wc))
        random.Random(seed).shuffle(list_files)
        # Divide list_files into smaller subsets
        #subset_size = 16
        #files_subsets = [list_files[i:i + subset_size] for i in range(0, len(list_files), subset_size)]
        #iterators_list = []
        #for files in files_subsets:
        dataset = tf.data.Dataset.list_files(list_files,seed=seed)
        dataset = tf.data.TFRecordDataset(dataset, compression_type='ZLIB', num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.with_options(options)

        dataset = dataset.map(
            lambda record: deserialize_tr(
                record,
                g, use_motif_activity,
                input_length, max_shift,
                output_length_ATAC, output_length,
                crop_size, output_res,
                atac_mask_dropout, random_mask_size,
                log_atac, use_atac, use_seq,
                atac_corrupt_rate),
            deterministic=False,
            num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.repeat(3).batch(batch).prefetch(tf.data.AUTOTUNE)

    else:
        list_files = (tf.io.gfile.glob(os.path.join(gcs_path, split, wc)))
        files = tf.data.Dataset.list_files(list_files,shuffle=False)
        dataset = tf.data.TFRecordDataset(files, compression_type='ZLIB', num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.with_options(options)
        dataset = dataset.map(
            lambda record: deserialize_val(
                record, g, use_motif_activity,
                input_length, max_shift,
                output_length_ATAC, output_length,
                crop_size, output_res,
                atac_mask_dropout, random_mask_size,
                log_atac, use_atac, use_seq),
            deterministic=False,
            num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.take(batch*validation_steps).batch(batch).repeat((num_epoch)).prefetch(tf.data.AUTOTUNE)

def return_distributed_iterators(gcs_path, gcs_path_ho, global_batch_size,
                                 input_length, max_shift, output_length_ATAC,
                                 output_length, crop_size, output_res,
                                 num_parallel_calls, num_epoch, strategy,
                                 options, options_val,
                                 atac_mask_dropout, atac_mask_dropout_val,
                                 random_mask_size,
                                 log_atac, use_atac, use_seq, seed,seed_val,
                                 atac_corrupt_rate, 
                                 validation_steps, use_motif_activity, g, g_val):

    tr_iterator = return_dataset(gcs_path, "train", global_batch_size, input_length,
                             output_length_ATAC, output_length, crop_size,
                             output_res, max_shift, options, num_parallel_calls,
                             num_epoch, atac_mask_dropout, random_mask_size,
                             log_atac, use_atac, use_seq, seed,
                             atac_corrupt_rate, validation_steps, use_motif_activity, g)

    val_data_ho = return_dataset(gcs_path_ho, "valid", global_batch_size, input_length,
                                 output_length_ATAC, output_length, crop_size,
                                 output_res, max_shift, options_val, num_parallel_calls, num_epoch,
                                 atac_mask_dropout_val, random_mask_size, log_atac,
                                 use_atac, use_seq, seed_val, atac_corrupt_rate,
                                 validation_steps, use_motif_activity, g_val)

    val_dist_ho=strategy.experimental_distribute_dataset(val_data_ho)
    val_data_ho_it = iter(val_dist_ho)

    #dist_iters_list=[]
    #for it in tr_iterators:
    tr_dist = strategy.experimental_distribute_dataset(tr_iterator)
    tr_data_it = iter(tr_dist)
    #    dist_iters_list.append(tr_data_it)

    return tr_data_it, val_data_ho_it

def early_stopping(current_val_loss,
                   logged_val_losses,
                   best_epoch,
                   patience,
                   patience_counter,
                   min_delta,):
    """early stopping function
    Args:
        current_val_loss: current epoch val loss
        logged_val_losses: previous epochs val losses
        current_epoch: current epoch number
        save_freq: frequency(in epochs) with which to save checkpoints
        patience: # of epochs to continue w/ stable/increasing val loss
                  before terminating training loop
        patience_counter: # of epochs over which val loss hasn't decreased
        min_delta: minimum decrease in val loss required to reset patience
                   counter
        model: model object
        save_directory: cloud bucket location to save model
        model_parameters: log file of all model parameters
        saved_model_basename: prefix for saved model dir
    Returns:
        stop_criteria: bool indicating whether to exit train loop
        patience_counter: # of epochs over which val loss hasn't decreased
        best_epoch: best epoch so far
    """
    print('check whether early stopping/save criteria met')
    try:
        best_loss = min(logged_val_losses[:-1])

    except ValueError:
        best_loss = current_val_loss

    stop_criteria = False
    ## if min delta satisfied then log loss

    if (current_val_loss >= (best_loss - min_delta)):# and (current_pearsons <= best_pearsons):
        patience_counter += 1
        if patience_counter >= patience:
            stop_criteria=True
    else:
        best_epoch = np.argmin(logged_val_losses)
        ## save current model

        patience_counter = 0
        stop_criteria = False

    return stop_criteria, patience_counter, best_epoch

def parse_args(parser):
    # Loads in command line arguments for execute_sweep.sh
    parser.add_argument('--tpu_name', help='name of TPU pod')
    parser.add_argument('--tpu_zone', help='zone of TPU pod')
    parser.add_argument('--wandb_project', help='name of wandb project to write to')
    parser.add_argument('--wandb_user', help='wandb username')
    parser.add_argument('--wandb_sweep_name', help='wandb_sweep_name')
    parser.add_argument('--gcs_project', help='gcs_project')
    parser.add_argument('--gcs_path', help='google bucket containing preprocessed data')
    parser.add_argument('--gcs_path_holdout', help='google bucket containing holdout data')
    parser.add_argument('--num_parallel', type=int, default=multiprocessing.cpu_count(), help='thread count for tensorflow record loading')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--val_examples_ho', type=int, help='val_examples_ho')
    parser.add_argument('--patience', type=int, help='patience for early stopping')
    parser.add_argument('--min_delta', type=float, help='min_delta for early stopping')
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--model_save_basename', type=str)
    parser.add_argument('--max_shift', default=10, type=int)
    parser.add_argument('--output_res', default=128, type=int)
    parser.add_argument('--decay_steps', default=88*34021*16, type=int)
    parser.add_argument('--lr_base', default="1.0e-03", help='lr_base')
    parser.add_argument('--decay_frac', type=str, help='decay_frac')
    parser.add_argument('--input_length', type=int, default=196608, help='input_length')
    parser.add_argument('--output_length', type=int, default=1536, help='output_length')
    parser.add_argument('--output_length_ATAC', type=int, default=1536, help='output_length_ATAC')
    parser.add_argument('--final_output_length', type=int, default=896, help='final_output_length')
    parser.add_argument('--num_transformer_layers', type=str, default="6", help='num_transformer_layers')
    parser.add_argument('--filter_list_seq', default="768,896,1024,1152,1280,1536", help='filter_list_seq')
    parser.add_argument('--filter_list_atac', default="32,64", help='filter_list_atac')
    parser.add_argument('--epsilon', default=1.0e-16, type=float, help='epsilon')
    parser.add_argument('--gradient_clip', type=str, default="1.0", help='gradient_clip')
    parser.add_argument('--dropout_rate', default="0.40", help='dropout_rate')
    parser.add_argument('--pointwise_dropout_rate', default="0.05", help='pointwise_dropout_rate')
    parser.add_argument('--num_heads', default="8", help='num_heads')
    parser.add_argument('--num_random_features', type=str, default="256", help='num_random_features')
    parser.add_argument('--BN_momentum', type=float, default=0.80, help='BN_momentum')
    parser.add_argument('--kernel_transformation', type=str, default="relu_kernel_transformation", help='kernel_transformation')
    parser.add_argument('--savefreq', type=int, help='savefreq')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint_path')
    parser.add_argument('--load_init', type=str, default="False", help='load_init')
    parser.add_argument('--normalize', type=str, default="True", help='normalize')
    parser.add_argument('--norm', type=str, default="True", help='norm')
    parser.add_argument('--atac_mask_dropout', type=float, default=0.05, help='atac_mask_dropout')
    parser.add_argument('--atac_mask_dropout_val', type=float, default=0.05, help='atac_mask_dropout_val')
    parser.add_argument('--final_point_scale', type=str, default="6", help='final_point_scale')
    parser.add_argument('--rectify', type=str, default="True", help='rectify')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer')
    parser.add_argument('--log_atac', type=str, default="True", help='log_atac')
    parser.add_argument('--use_atac', type=str, default="True", help='use_atac')
    parser.add_argument('--use_seq', type=str, default="True", help='use_seq')
    parser.add_argument('--random_mask_size', type=str, default="1152", help='random_mask_size')
    parser.add_argument('--seed', type=int, default=42, help= 'seed')
    parser.add_argument('--val_data_seed', type=int, default=25, help= 'val_data_seed')
    parser.add_argument('--atac_corrupt_rate', type=str,default="20",
                        help= 'increase atac corrupt by 3x with 1.0/atac_corrupt_rate probability')
    parser.add_argument('--use_motif_activity', type=str, default="False", help= 'use_motif_activity')
    parser.add_argument('--loss_type', type=str, default="poisson_multinomial", help= 'loss_type')
    parser.add_argument('--total_weight_loss',type=str, default="0.15", help= 'total_weight_loss')
    parser.add_argument('--use_rot_emb',type=str, default="True", help= 'use_rot_emb')
    parser.add_argument('--best_val_loss', type=float, default=0.09113)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--warmup_frac', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--reset_optimizer_state',type=str, default="False", help= 'reset_optimizer_state')
    parser.add_argument('--return_constant_lr',type=str, default="False", help= 'return_constant_lr')
    args = parser.parse_args()
    return parser


def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=4)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters),
                      depth = 5,
                      dtype=tf.float32)[:, :4]
    return out


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def tf_tpu_initialize(tpu_name,zone):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """

    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name,zone=zone)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy
'''
def make_plots(y_trues,
               y_preds,
               num_points):

    results_df = pd.DataFrame()
    results_df['true'] = y_trues
    results_df['pred'] = y_preds

    results_df['true_log'] = np.log2(1.0+results_df['true'])
    results_df['pred_log'] = np.log2(1.0+results_df['pred'])

    true=results_df[['true']].to_numpy()[:,0]
    pred=results_df[['pred']].to_numpy()[:,0]
    true_log=results_df[['true_log']].to_numpy()[:,0]
    pred_log=results_df[['pred_log']].to_numpy()[:,0]

    try:
        overall_corr=results_df['true'].corr(results_df['pred'])
        overall_corr_log=results_df['true_log'].corr(results_df['pred_log'])
    except np.linalg.LinAlgError as err:
        overall_corr = 0.0
        overall_corr_log = 0.0

    fig_overall,ax_overall=plt.subplots(figsize=(6,6))
    ## scatter plot for 50k points max
    idx = np.random.choice(np.arange(len(true)), num_points, replace=False)

    data = np.vstack([true[idx], pred[idx]])

    min_true = min(true)
    max_true = max(true)
    min_pred = min(pred)
    max_pred = max(pred)

    try:
        kernel = stats.gaussian_kde(data)(data)
        sns.scatterplot(
            x=true_log[idx],
            y=pred_log[idx],
            c=kernel,
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall atac corr")
    except np.linalg.LinAlgError as err:
        sns.scatterplot(
            x=true_log[idx],
            y=pred_log[idx],
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall atac corr")
    except ValueError:
        sns.scatterplot(
            x=true_log[idx],
            y=pred_log[idx],
            cmap="viridis")
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall atac corr")

    return fig_overall, overall_corr,overall_corr_log
'''

def mask_ATAC_profile(output_length_ATAC, output_length, crop_size, mask_size,output_res, 
                        peaks_c_crop, randomish_seed, atac_mask_int, atac_mask_dropout):
    """
    Creates a random mask for the input ATAC profile
    The mask consists of 1s and 0s, where 1s indicate masked regions.
    Mask is generated by selecting one peak to mask in the window if present,
    and randomly masking atac_mask_dropout % of the remaining bins

    Parameters:
    - output_length_ATAC: the length of the input ATAC profile (at 4 bp resolution)
    - output_length: The length of the output ATAC profile
    - crop_size: The size of the cropped region to take from each edge of output ATAC profile
    - mask_size: The size of the masked region in basepairs
    - output_res: The resolution of the output target ATAC profile
    - peaks_c_crop: The center of the peaks in the output ATAC profile
    - randomish_seed: A random seed to use for the random masking
    - atac_mask_dropout: The dropout rate for the ATAC masking
    Returns:
    - The mask as well as inverted mask. Inverted mask is used to set the masked regions to 0. 
    """

    num_mask_bins = mask_size // output_res ## calculate the number of bins that will be masked in each region
    out_length_cropped = output_length-2*crop_size
    if out_length_cropped % num_mask_bins != 0:
        raise ValueError('ensure that masking size divided by output res is a factor of the cropped output length')
    center = (output_length-2*crop_size)//2 # compute center of the output window

    
    # -------------------------- here we set up masking one of the peaks ----------------------------------------
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0] # get indices of peak centers, randomly select one
    mask_indices_temp = tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5])

    if tf.size(mask_indices_temp) > 0: # in the case where we did actually have peaks in the window
        ridx = tf.concat([mask_indices_temp],axis=0) # adjust dimensions by concatenating selected indices
        start_index = ridx[0] - num_mask_bins // 2 + crop_size # compute start index of masked PEAK region
        end_index = ridx[0] + 1 + num_mask_bins // 2 + crop_size # compute end index of masked PEAK region
        indices = tf.range(start_index, end_index) # get range of indices
        mask = (indices >= 0) & (indices < output_length) # mask to apply to indices to ensure they are valid
        filtered_indices = tf.boolean_mask(indices, mask) # apply index mask
        mask_indices = tf.cast(tf.reshape(filtered_indices, [-1, 1]), dtype=tf.int64) # now we get the full indices
        st=tf.SparseTensor( # create sparse tensor with 1s at masked indices
            indices=mask_indices,
            values=tf.ones([tf.shape(mask_indices)[0]], dtype=tf.float32),
            dense_shape=[output_length])
        dense_peak_mask=tf.sparse.to_dense(st)
        dense_peak_mask=1.0-dense_peak_mask # invert the mask to actually use
    else:
        dense_peak_mask = tf.ones([output_length],dtype=tf.float32) # if no peaks in window, no mask
    dense_peak_mask = tf.expand_dims(dense_peak_mask,axis=1) # adjust dimenions to multiply by ATAC profile

    # -----------------------here we set up the RANDOM mask (not peak based) ------------------------------------
    edge_append = tf.ones((crop_size,1),dtype=tf.float32) # we don't want the mask on the edge, so set these to 1

    # we will create the mask by first initializing a tensor of 1s, with shape 
    # based on the output cropped length and the number of bins to mask in each region. 
    # e.g., if we want to mask 
    atac_mask = tf.ones(out_length_cropped // num_mask_bins,dtype=tf.float32)

    if ((atac_mask_int == 0)): # with 1/atac_mask_int probability, increase random masking percentage 
        atac_mask_dropout = 3 * atac_mask_dropout
    
    # here create the random mask using dropout 
    atac_mask=tf.nn.experimental.stateless_dropout(atac_mask,
                                              rate=(atac_mask_dropout),
                                              seed=[0,randomish_seed-5]) / (1. / (1.0-(atac_mask_dropout)))
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask = tf.tile(atac_mask, [1,num_mask_bins])
    atac_mask = tf.reshape(atac_mask, [-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    full_atac_mask = tf.concat([edge_append,atac_mask,edge_append],axis=0)


    # -----------------------here we COMBINE atac and peak mask -------------------------------------------------
    full_comb_mask = tf.math.floor((dense_peak_mask + full_atac_mask)/2) # if either mask is 0, mask value set to 0
    full_comb_mask_store = 1.0 - full_comb_mask # store the mask after inverting it to get the masked region indices

    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)

    if crop_size > 0:
        full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:] # store the cropped mask

    return full_comb_mask, full_comb_mask_store


def mask_sequence(sequence, sequence_length, bin_size=32, kmer_size=1, seed=4):
    num_stretches = sequence_length // bin_size

    start_indices = tf.random.stateless_uniform(shape=[num_stretches], 
                                                minval=0,
                                                maxval=bin_size - kmer_size + 1,
                                                seed=[seed+11, seed+2],
                                                dtype=tf.int32) + \
                    tf.range(0, sequence_length, bin_size)
    # Use broadcasting to add range_tensor to each element in start_indices
    all_indices_tensor = start_indices[:, tf.newaxis] + tf.range(kmer_size)
    all_indices_tensor = tf.expand_dims(tf.reshape(all_indices_tensor,[-1]),
                                        axis=1)

    mask = tf.ones(sequence_length)
    updates = tf.squeeze(tf.zeros_like(all_indices_tensor,dtype=tf.float32))
    mask = tf.tensor_scatter_nd_update(mask, all_indices_tensor, updates)

    sequence_masked = sequence * tf.expand_dims(mask,axis=1)
    return sequence_masked





'''
simplify

def mask_ATAC_profile(output_length_ATAC, output_length, crop_size, mask_size,output_res, 
                        peaks_c_crop, randomish_seed, atac_mask_int, atac_mask_dropout):
    """
    Creates a random mask for the input ATAC profile
    The mask consists of 1s and 0s, where 1s indicate masked regions.
    Mask is generated by selecting one peak to mask in the window if present,
    and randomly masking atac_mask_dropout % of the remaining bins

    Parameters:
    - output_length_ATAC: the length of the input ATAC profile (at 4 bp resolution)
    - output_length: The length of the output ATAC profile
    - crop_size: The size of the cropped region to take from each edge of output ATAC profile
    - mask_size: The size of the masked region in basepairs
    - output_res: The resolution of the output target ATAC profile
    - peaks_c_crop: The center of the peaks in the output ATAC profile
    - randomish_seed: A random seed to use for the random masking
    - atac_mask_dropout: The dropout rate for the ATAC masking
    Returns:
    - The mask as well as inverted mask. Inverted mask is used to set the masked regions to 0. 
    """

    num_mask_bins = mask_size // output_res ## calculate the number of bins that will be masked in each region
    out_length_cropped = output_length-2*crop_size
    if out_length_cropped % num_mask_bins != 0:
        raise ValueError('ensure that masking size divided by output res is a factor of the cropped output length')
    center = (output_length-2*crop_size)//2 # compute center of the output window

    
    # -------------------------- here we set up masking one of the peaks ----------------------------------------
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0] # get indices of peak centers, randomly select one
    mask_indices_temp = tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5])

    if tf.size(mask_indices_temp) > 0: # in the case where we did actually have peaks in the window
        ridx = tf.concat([mask_indices_temp],axis=0) # adjust dimensions by concatenating selected indices
        start_index = ridx[0] - num_mask_bins // 2 + crop_size # compute start index of masked PEAK region
        end_index = ridx[0] + 1 + num_mask_bins // 2 + crop_size # compute end index of masked PEAK region
        indices = tf.range(start_index, end_index) # get range of indices
        mask = (indices >= 0) & (indices < output_length) # mask to apply to indices to ensure they are valid
        filtered_indices = tf.boolean_mask(indices, mask) # apply index mask
        mask_indices = tf.cast(tf.reshape(filtered_indices, [-1, 1]), dtype=tf.int64) # now we get the full indices
        st=tf.SparseTensor( # create sparse tensor with 1s at masked indices
            indices=mask_indices,
            values=tf.ones([tf.shape(mask_indices)[0]], dtype=tf.float32),
            dense_shape=[output_length])
        dense_peak_mask=tf.sparse.to_dense(st)
        dense_peak_mask=1.0-dense_peak_mask # invert the mask to actually use
    else:
        dense_peak_mask = tf.ones([output_length],dtype=tf.float32) # if no peaks in window, no mask
    dense_peak_mask = tf.expand_dims(dense_peak_mask,axis=1) # adjust dimenions to multiply by ATAC profile

    # -----------------------here we set up the RANDOM mask (not peak based) ------------------------------------
    edge_append = tf.ones((crop_size,1),dtype=tf.float32) # we don't want the mask on the edge, so set these to 1

    # we will create the mask by first initializing a tensor of 1s, with shape 
    # based on the output cropped length and the number of bins to mask in each region. 
    # e.g., if we want to mask 
    atac_mask = tf.ones(out_length_cropped // num_mask_bins,dtype=tf.float32)

    if ((atac_mask_int == 0)): # with 1/atac_mask_int probability, increase random masking percentage 
        atac_mask_dropout = 3 * atac_mask_dropout
    
    # here create the random mask using dropout 
    atac_mask=tf.nn.experimental.stateless_dropout(atac_mask,
                                              rate=(atac_mask_dropout),
                                              seed=[0,randomish_seed-5]) / (1. / (1.0-(atac_mask_dropout)))
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask = tf.tile(atac_mask, [1,num_mask_bins])
    atac_mask = tf.reshape(atac_mask, [-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    full_atac_mask = tf.concat([edge_append,atac_mask,edge_append],axis=0)

    # -----------------------here we COMBINE atac and peak mask -------------------------------------------------
    full_comb_mask = tf.math.floor((dense_peak_mask + full_atac_mask)/2) # if either mask is 0, mask value set to 0
    full_comb_mask_store = 1.0 - full_comb_mask # store the mask after inverting it to get the masked region indices

    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)

    if crop_size > 0:
        full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:] # store the cropped mask

    return full_comb_mask, full_comb_mask_store
'''