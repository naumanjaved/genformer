import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math 
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
from datetime import datetime
import random

import multiprocessing
#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import sonnet as snt
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

import pandas as pd
import seaborn as sns

from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import linregress
from scipy import stats
import keras.backend as kb

import scipy.special
import scipy.stats
import scipy.ndimage

import metrics
from scipy.stats import zscore

tf.keras.backend.set_floatx('float32')

def tf_tpu_initialize(tpu_name):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """
    
    try: 
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy


"""
having trouble w/ providing organism/step inputs to train/val steps w/o
triggering retracing/metadata resource exhausted errors, so defining 
them separately for hg, mm 
to do: simplify to two functions w/ organism + mini_batch_step_inputs
consolidate into single simpler function
"""


def return_test_build_functions(model,
                               strategy,
                               metric_dict):
    """Returns distributed test function
    Args:
        model: model object
    Returns:
        distributed test function
    
    return distributed train and val step functions for given organism
    test_steps
    """

    metric_dict["hg_test"] = tf.keras.metrics.Mean("hg_test_loss",
                                                  dtype=tf.float32)
    metric_dict['pearsonsR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)

    @tf.function(jit_compile=True)
    def test_step(inputs):
        target=tf.cast(inputs['target'],
                       dtype = tf.float32)
        target_rev=tf.cast(inputs['target_rev'],
                       dtype = tf.float32)
        
        
        target_mean = (target + tf.reverse(target, axis=[1]))/2.0
        
        
        sequence=tf.cast(inputs['sequence'],
                         dtype=tf.float32)
        rev_comp_sequence=tf.cast(inputs['rev_comp_sequence'],
                         dtype=tf.float32)
        tss_mask =tf.cast(inputs['tss_mask'],dtype=tf.float32)
        tss_mask_rev =tf.cast(inputs['tss_mask_rev'],dtype=tf.float32)
        
        output = tf.cast(model(sequence, is_training=False)['human'],
                         dtype=tf.float32)
        
        output_rev = tf.cast(model(rev_comp_sequence, is_training=False)['human'],
                         dtype=tf.float32)
        output_mean = (output + tf.reverse(output_rev,axis=[1]))/2.0
        
        pred = tf.reduce_sum(output * tss_mask,axis=1)
        true = tf.reduce_sum(target * tss_mask,axis=1)
        
        pred_rev = tf.reduce_sum(output_rev * tss_mask_rev,axis=1)
        true_rev = tf.reduce_sum(target_rev * tss_mask_rev,axis=1)
        
        gene_name = tf.cast(inputs['gene_name'],dtype=tf.int32)
        cell_types = tf.cast(inputs['cell_types'],dtype=tf.int32)
        
        
        metric_dict['pearsonsR'].update_state(target_mean, output_mean)
        metric_dict['R2'].update_state(target_mean, output_mean)
        
        pred_mean = (pred + pred_rev)/2.0
        true_mean = (true + true_rev)/2.0

        return pred_mean, true_mean, gene_name, cell_types

        
    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            target=tf.cast(inputs['target'],
                           dtype = tf.float32)
            sequence=tf.cast(inputs['sequence'],
                             dtype=tf.float32)
            output = model(sequence, is_training=False)['human']

        for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step, args=(next(iterator),))
        
    return test_step, build_step, metric_dict

        
def deserialize_val_TSS(serialized_example,input_length=196608,max_shift=4, out_length=1536,num_targets=50):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
        'tss_mask': tf.io.FixedLenFeature([], tf.string),
        'gene_name': tf.io.FixedLenFeature([], tf.string)
    }

    shift = 2
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + max_shift, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [shift,0],[input_length,-1])
    
    rev_comp_sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
    rev_comp_sequence = tf.reverse(rev_comp_sequence, axis=[0])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (out_length, num_targets))
    target = tf.slice(target,
                      [320,0],
                      [896,-1])
    target = tf.math.pow(target,0.50)
    target_rev = tf.reverse(target,axis=[0])
    
    tss_mask = tf.io.parse_tensor(example['tss_mask'],
                                  out_type=tf.int32)
    tss_mask = tf.slice(tss_mask,
                      [320,0],
                      [896,-1])
    tss_mask_rev = tf.reverse(tss_mask_rev,axis=[0])
    gene_name= tf.io.parse_tensor(example['gene_name'],out_type=tf.int32)
    gene_name = tf.tile(tf.expand_dims(gene_name,axis=0),[50])
    
    cell_types = tf.range(0,50)

    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'rev_comp_sequence': tf.ensure_shape(rev_comp_sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,50]),
            'target_rev': tf.ensure_shape(target_rev,
                                      [896,50]),
            'tss_mask': tf.ensure_shape(tss_mask,
                                        [896,1]),
            'tss_mask_rev': tf.ensure_shape(tss_mask_rev,
                                        [896,1]),
            'gene_name': tf.ensure_shape(gene_name,
                                         [50,]),
            'cell_types': tf.ensure_shape(cell_types,
                                           [50,])}
                    
def return_dataset(gcs_path,
                   split,
                   tss_bool,
                   batch,
                   input_length,
                   max_shift,
                   out_length,
                   num_targets,
                   options,
                   num_parallel):

    """
    return a tf dataset object for given gcs path
    """
    wc = str(split) + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize_val_TSS(record,
                                                     input_length,
                                                     max_shift,
                                                     out_length,
                                                     num_targets),
                          deterministic=False,
                          num_parallel_calls=num_parallel)
    
    
    dataset_build = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset_build = dataset_build.with_options(options)

    dataset_build = dataset_build.map(lambda record: deserialize_val_TSS(record,
                                                     input_length,
                                                     max_shift,
                                                     out_length,
                                                     num_targets),
                          deterministic=False,
                          num_parallel_calls=num_parallel)

    return dataset.batch(batch).prefetch(tf.data.AUTOTUNE).repeat(2), \
            dataset_build.batch(batch).prefetch(tf.data.AUTOTUNE).repeat(2)



def return_distributed_iterators(gcs_path_tss,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 out_length,
                                 num_targets,
                                 num_parallel_calls,
                                 strategy,
                                 options):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
    with strategy.scope():
        val_data_TSS = return_dataset(gcs_path_tss,
                                 "test",
                                 True,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 out_length,
                                 num_targets,
                                 options,
                                 num_parallel_calls)
            
        val_dist_TSS= strategy.experimental_distribute_dataset(val_data_TSS)

        val_data_TSS_it = iter(val_dist_TSS)


    return val_data_TSS_it


def make_plots(y_trues,
               y_preds, 
               cell_types, 
               gene_map):

    results_df = pd.DataFrame()
    results_df['true'] = y_trues
    results_df['pred'] = y_preds
    results_df['cell_type_encoding'] = cell_types
    results_df['gene_encoding'] = gene_map
    
    results_df=results_df.groupby(['gene_encoding', 'cell_type_encoding']).agg({'true': 'sum', 'pred': 'sum'})
    results_df['true_log1p'] = np.log2(1.0+results_df['true'])
    results_df['pred_log1p'] = np.log2(1.0+results_df['pred'])
    
    #results_df['true_zscore'] = df.groupby('cell_type_encoding')['true'].apply(lambda x: (x - x.mean())/x.std())
    results_df['true_zscore']=results_df.groupby(['cell_type_encoding']).true_log1p.transform(lambda x : zscore(x))
    #results_df['pred_zscore'] = df.groupby('cell_type_encoding')['pred'].apply(lambda x: (x - x.mean())/x.std())
    results_df['pred_zscore']=results_df.groupby(['cell_type_encoding']).pred_log1p.transform(lambda x : zscore(x))
    
    true_zscore=results_df[['true_zscore']].to_numpy()[:,0]

    pred_zscore=results_df[['pred_zscore']].to_numpy()[:,0]

    try: 
        cell_specific_corrs=results_df.groupby('cell_type_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
        cell_specific_corrs_raw=results_df.groupby('cell_type_encoding')[['true','pred']].corr(method='pearson').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        cell_specific_corrs = [0.0] * len(np.unique(cell_types))

    try: 
        gene_specific_corrs=results_df.groupby('gene_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
        gene_specific_corrs_raw=results_df.groupby('gene_encoding')[['true','pred']].corr(method='pearson').unstack().iloc[:,1].tolist()

    except np.linalg.LinAlgError as err:
        gene_specific_corrs = [0.0] * len(np.unique(gene_map))
    
    corrs_overall = np.nanmean(cell_specific_corrs), np.nanmean(gene_specific_corrs), \
                        np.nanmean(cell_specific_corrs_raw), np.nanmean(gene_specific_corrs_raw)
                        
    return corrs_overall, results_df

def parse_args(parser):
    """Loads in command line arguments
    """
        
    parser.add_argument('--tpu_name', dest = 'tpu_name',
                        help='tpu_name')
    parser.add_argument('--tpu_zone', dest = 'tpu_zone',
                        help='tpu_zone')
    parser.add_argument('--wandb_project', 
                        dest='wandb_project',
                        help ='wandb_project')
    parser.add_argument('--wandb_user',
                        dest='wandb_user',
                        help ='wandb_user')
    parser.add_argument('--wandb_sweep_name',
                        dest='wandb_sweep_name',
                        help ='wandb_sweep_name')
    parser.add_argument('--gcs_project', dest = 'gcs_project',
                        help='gcs_project')
    parser.add_argument('--gcs_path_TSS',
                        dest='gcs_path_TSS',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=tf.data.AUTOTUNE,
                        help='thread count for tensorflow record loading')
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--input_length',
                        dest='input_length',
                        default=196608,
                        type=int,
                        help='input_length')
    parser.add_argument('--num_targets',
                        dest='num_targets',
                        type=int,
                        default=50,
                        help= 'num_targets')
    parser.add_argument('--test_examples', dest = 'test_examples',
                        type=int, help='test_examples')
    parser.add_argument('--checkpoint_path', dest = 'checkpoint_path',
                        help='checkpoint_path',
                        default=None)
    
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
    table = tf.lookup.StaticHashTable(init, default_value=0)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out

def rev_comp_one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))
    input_characters = tf.reverse(input_characters,[0])
    
    vocabulary = tf.constant(['T', 'G', 'C', 'A'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out



def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


    
    
    
