#!/usr/bin/env python3

import argparse
import collections
import gzip
import math
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pybedtools as pybt

import tensorflow as tf
import time

from scipy import stats
from scipy.signal import find_peaks

import cython
import pyximport; pyximport.install(reload_support=True)
import cython_fxn
pd.options.mode.chained_assignment = None 

'''
sequence processing utilities
'''
################################################################################
def installed(name):
    '''
    Checks whether program is installed

    Return:
        - true if installed and in path
    '''
    return shutil.which(name) is not None

def interval_to_bed(interval):
    '''
    Convert pybedtools interval object to bed object
    '''
    tab = '\t'

    interval_str = tab.join([interval.chrom, str(interval.start),
                                            str(interval.stop)])

    return pybt.BedTool(interval_str, from_string=True)
################################################################################
def valid_chrom(chrom):
    accepted_chroms = ['chr' + str(i) for i in range(23)]
    
    if chrom not in accepted_chroms:
         return False
    else:
         return True
################################################################################
def compute_intervals(length, stride, genome_file):#, blacklist):
    '''
    Compute intervals of desired size excluding ENCODE ATAC blacklist sites

    Return:
        - intervals bed object
    '''

    if not installed("bedtools"):
        raise EnvironmentError('bedtools not found in path')

    ## ensure blacklist is sorted
    #blacklist_bed = pybt.BedTool(blacklist).sort()

    #mappable = blacklist_bed.complement(g=genome_file)

    intervals = pybt.BedTool().window_maker(g=genome_file,
                                        w=length,
                                        s=stride)
    ## ensure proper length
    intervals = intervals.filter(lambda x: len(x) == length)

    return intervals
################################################################################
def compute_TSS_intervals(length,
                          tss_sites_file,
                          genome_file):#, blacklist):
    '''
    Compute intervals of desired size excluding ENCODE ATAC blacklist sites

    Return:
        - intervals bed object
    '''
    tss_sites = pd.read_csv(tss_sites_file, sep = '\t')
    
    chrom_sizes = pd.read_csv(genome_file, sep = '\t')
    chrom_sizes.columns = ['chrom', 'size']
    
    tss_sites = tss_sites.merge(chrom_sizes, left_on='seqnames', right_on='chrom')
    
    ### expand intervals
    tss_sites['tss_start'] = tss_sites['rep_TSS'] - (length // 2 - 1)
    tss_sites['tss_end'] = tss_sites['rep_TSS'] + (length // 2 + 1)
    
    tss_sites = tss_sites.loc[tss_sites['tss_start'] >= 0]
    tss_sites = tss_sites.loc[tss_sites['tss_end'] < tss_sites['size']]
    
    intervals_df = tss_sites[["seqnames", "tss_start", "tss_end", "gene_id"]]
    
    intervals_df.columns = ["chrom", "start", "stop", "name"]
    intervals = pybt.BedTool.from_dataframe(intervals_df)
    
    ## ensure proper length
    intervals = intervals.filter(lambda x: len(x) == length)
    #print(intervals_df)

    return intervals
################################################################################
def return_sequence(interval, genome_fasta):
    '''
    Extract sequence for target intervals

    Return

    '''
    tab = '\t'
    interval_bed = interval_to_bed(interval)

    #interval_bed = interval_bed.sequence(fi=genome_fasta)
    interval_str = interval.chrom + ':' + \
                    str(interval.start + 1) + \
                    '-' + str(interval.stop)
    return interval_bed.seq(interval_str, genome_fasta)



################################################################################
def process_bedgraph(interval, df):
    '''
    Extract coverage info for input pybedtools interval object

    Return:
        numpy array of interval length where each pos is 5' read coverage
    '''
    
    df['start'] = df['start'].astype('int64') - int(interval.start)
    df['end'] = df['end'].astype('int64') - int(interval.start)
    
    ## ensure within bounds of interval
    df.loc[df.start < 0, 'start'] = 0
    df.loc[df.end > len(interval), 'end'] = len(interval)
    
    per_base = np.zeros(len(interval), dtype=np.float64)

    num_intervals = df['start'].to_numpy().shape[0]

    output = np.asarray(cython_fxn.get_per_base_score_f(
                    df['start'].to_numpy().astype(np.int_),
                    df['end'].to_numpy().astype(np.int_),
                    df['score'].to_numpy().astype(np.float64),
                    per_base, 
                    num_intervals), dtype = np.float64)
    ### bin at the desired output res
    return output

def concatenate(one_hot_sequence, bedgraph):
    '''
    take in one hot encoded sequence 
    take in 1D array of bedgraph coverage scores at each position 
    '''
    assert one_hot_sequence.shape[0] == bedgraph.shape[0]
    
    return np.column_stack((one_hot_sequence, bedgraph))


def process_bedgraph_binned(interval, bedgraph, output_res):
    '''
    Extract coverage info over input pybedtools interval in bins of desired size

    Return:
        - np array of size interval // ouput_res with summed 5' read coverage

    '''
    output_bedgraph = process_bedgraph(interval, bedgraph)

    num_bins = len(interval) // output_res

    def bin_vector(input_arr, num_bins, output_res):
        return np.sum(input_arr.reshape(num_bins, output_res), axis=1)
        #return summed, mean, var
    
    return bin_vector(output_bedgraph, num_bins, output_res)

def one_hot(sequence, replaceN = 'all_1'):
    '''
    Convert input sequence to one hot encoded numpy array. Not used

    Return:
        - np array of one hot encoded sequence
    '''

    if replaceN not in ['all_1', 'all_0', 'random']:
        raise ValueError('N_rep must be one of all_1, all_0, random')
    
    np_sequence = np.array(list(sequence.upper()))
    ## initialize empty numpy array
    length = len(sequence)
    one_hot_out = np.zeros((length, 4))

    one_hot_out[np_sequence == 'A'] = [1, 0, 0, 0]
    one_hot_out[np_sequence == 'T'] = [0, 1, 0, 0]
    one_hot_out[np_sequence == 'C'] = [0, 0, 1, 0]
    one_hot_out[np_sequence == 'G'] = [0, 0, 0, 1]

    replace = 4 * [0]
    if replaceN == 'all_0':
        replace = 4 * [1]
    if replaceN == 'random':
        rand = np.random.randint(4, size = 1)[0]
        replace[rand] = 1
    one_hot_out[np_sequence == 'N'] = replace

    return one_hot_out
    

def soft_clip(input_array, clip_threshold):
    clip = np.int64(clip_threshold)
    clipped = clip + np.floor(np.sqrt(abs(input_array - clip)))
    arr = np.array([input_array, clipped])
    
    return arr.min(axis=0)

def get_exon_tokens(interval,
                    gene_name,
                    exons_df):
    

    exons_df_sub = exons_df.loc[exons_df['gene_IDs'] == gene_name]
    exons_df_sub['score'] = 1
    
    ## ensure within bounds of interval
    exons_df_sub.loc[exons_df_sub.start < 0, 'start'] = 0
    exons_df_sub.loc[exons_df_sub.end > len(interval), 'end'] = len(interval)

    per_base = np.zeros(len(interval), dtype=np.int_)
    
    num_intervals = exons_df_sub['start'].to_numpy().shape[0]
    
    output = np.asarray(cython_fxn.get_per_base_cov(
                    exons_df_sub['start'].to_numpy().astype(np.int_),
                    exons_df_sub['end'].to_numpy().astype(np.int_),
                    exons_df_sub['score'].to_numpy().astype(np.int_),
                    per_base, 
                    num_intervals), dtype = np.int64)
    
    return output


def get_generic_token(interval,
                      input_df):
    
    input_df['score'] = 1
    
    ## ensure within bounds of interval
    input_df.loc[input_df.start < 0, 'start'] = 0
    input_df.loc[input_df.end > len(interval), 'end'] = len(interval)

    per_base = np.zeros(len(interval), dtype=np.int_)
    
    num_intervals = input_df['start'].to_numpy().shape[0]
    
    output = np.asarray(cython_fxn.get_per_base_cov(
                    input_df['start'].to_numpy().astype(np.int_),
                    input_df['end'].to_numpy().astype(np.int_),
                    input_df['score'].to_numpy().astype(np.int_),
                    per_base, 
                    num_intervals), dtype = np.int64)
    return output

def process_quants(interval,
                   gene,
                   gene_quants):
    
    sub_df = gene_quants.loc[gene_quants['ID'] == gene]['TPM'].to_list()
    #sub_df_uqn = gene_quants.loc[gene_quants['ID'] == gene]['TPM_uqn'].to_list()
    if len(sub_df) == 0:
        return -1
    if len(sub_df) > 1:
        print('repeated genes....')
     
    return sub_df[0]#, sub_df_uqn[0]



def check_center_empty(input_array):
    length = input_array.shape[0]
    start_index = 1 * length // 4
    end_index = 3 * length // 4
    
    sub_bin = input_array[start_index: end_index]
    
    return (np.sum(sub_bin) == 0.0)
    
    
def get_TSS_token(interval, input_df, num_bins, output_res, shift_amt): #gene_name, 
    input_df['ann_token'] = 1
    
    per_base = np.zeros(len(interval), dtype=np.int_)
    
    num_intervals = input_df['start'].to_numpy().shape[0]

    output = np.asarray(cython_fxn.get_per_base_cov(
                    input_df['start'].to_numpy().astype(np.int_),
                    input_df['end'].to_numpy().astype(np.int_),
                    input_df['ann_token'].to_numpy().astype(np.int_),
                    per_base, 
                    num_intervals), dtype = np.int64)

    
    output = output[shift_amt:-shift_amt]

    return np.max(output.reshape(num_bins, output_res), axis=1)
    
def get_gene_token(interval, input_df, num_bins, output_res, shift_amt): #gene_name

    #input_df_sub = input_df.loc[input_df['gene_IDs'] == gene_name]

    ## ensure within bounds of interval
    #input_df.loc[input_df.start < 0, 'start'] = 0
    #input_df.loc[input_df.end > len(interval), 'end'] = len(interval)
    
    per_base = np.zeros(len(interval), dtype=np.int_)
    
    num_intervals = input_df['start'].to_numpy().shape[0]

    output = np.asarray(cython_fxn.get_per_base_cov(
                    input_df['start'].to_numpy().astype(np.int_),
                    input_df['end'].to_numpy().astype(np.int_),
                    input_df['gene_encoded'].to_numpy().astype(np.int_),
                    per_base, 
                    num_intervals), dtype = np.int64)
    output = output[shift_amt:-shift_amt]
    return np.max(output.reshape(num_bins, output_res), axis=1)

def parse_cell_map(cell_map_file):
    input_df = pd.read_csv(cell_map_file,sep='\t',header=None)
    
    input_df.columns = ['cell_type', 'index']
    
    cell_types = input_df['cell_type'].tolist()
    index = input_df['index'].tolist()
    
    return {cell_t: index[k] for k,cell_t in enumerate(cell_types)}

def parse_interval_map(intervals_file):
    subprocess.call("sort -k1,1 -k2,2n " + intervals_file + " > intervals.sort.bed",
                    shell=True)
    input_df = pd.read_csv("intervals.sort.bed", sep='\t',header=None)
    
    input_df.columns = ['chrom', 'start',
                         'end', 'subset']
    
    intervals_lookup_dict={}
    for k,row in input_df.iterrows():
        interval = '\t'.join([row['chrom'],
                              str(row['start']),
                              str(row['end'])])
        intervals_lookup_dict[interval]=k

    return intervals_lookup_dict