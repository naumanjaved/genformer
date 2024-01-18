import tensorflow as tf
import tf.keras.initializers as inits
import tensorstore as ts
import numpy as np

def load(b,path):
    array = ts.open(
        {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': os.path.join(b,path)
            },
        })
    result = array.result().read().result()
    return result

def get_initializers_genformer_ft(checkpoint_path, num_transformer_layers=6):

    b= checkpoint_path
    initializers_dict = {'stem_conv_k': inits.Constant(load(b,'target.EnformerConvolutions_0.Conv_0.kernel')),
                         'stem_conv_b': inits.Constant(load(b,'target.EnformerConvolutions_0.Conv_1.bias')),
                         'stem_res_conv_k': inits.Constant(load(b,'target.EnformerConvolutions_0.Conv_1.kernel')),
                         'stem_res_conv_b': inits.Constant(load(b,'target.EnformerConvolutions_0.Conv_1.bias')),
                         'stem_res_conv_BN_g': inits.Constant(load(b,'target.EnformerConvolutions_0.BatchNorm_0.scale')),
                         'stem_res_conv_BN_b': inits.Constant(load(b,'target.EnformerConvolutions_0.BatchNorm_0.bias')),
                         'stem_res_conv_BN_m': inits.Constant(load(b,'flax_mutables.batch_stats.EnformerConvolutions_0.BatchNorm_0.mean')),
                         'stem_res_conv_BN_v': inits.Constant(load(b,'flax_mutables.batch_stats.EnformerConvolutions_0.BatchNorm_0.var'))}

    temp_dict = {'stem_conv_atac_k': inits.Constant(load(b,'target.EnformerConvolutions_1.Conv_0.kernel')),
                         'stem_conv_atac_b': inits.Constant(load(b,'target.EnformerConvolutions_1.Conv_0.bias')),
                         'stem_res_conv_atac_k': inits.Constant(load(b,'target.EnformerConvolutions_1.Conv_1.kernel')),
                         'stem_res_conv_atac_b': inits.Constant(load(b,'target.EnformerConvolutions_1.Conv_1.bias')),
                         'stem_res_conv_atac_BN_g': inits.Constant(load(b,'target.EnformerConvolutions_1.BatchNorm_0.scale')),
                         'stem_res_conv_atac_BN_b': inits.Constant(load(b,'target.EnformerConvolutions_1.BatchNorm_0.bias')),
                         'stem_res_conv_atac_BN_m': inits.Constant(load(b,'flax_mutables.batch_stats.EnformerConvolutions_1.BatchNorm_0.mean')),
                         'stem_res_conv_atac_BN_v': inits.Constant(load(b,'flax_mutables.batch_stats.EnformerConvolutions_1.BatchNorm_0.var'))}
    initializers_dict.update(temp_dict)

    temp_dict = {'final_point_k': inits.Constant(load(b,'target.final_pointwise.kernel')),
                         'final_point_b': inits.Constant(load(b,'target.final_pointwise.bias'))}
                         #'final_point_BN_g': inits.Constant(load(b,'final_pointwise_conv/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         #'final_point_BN_b': inits.Constant(load(b,'final_pointwise_conv/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         #'final_point_BN_m': inits.Constant(load(b,'final_pointwise_conv/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         #'final_point_BN_v': inits.Constant(load(b,'final_pointwise_conv/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(temp_dict)

    out_dict = {'motif_activity_fc1_k': inits.Constant(load(b,'target.projection_tf1_in.kernel'))}
                #'motif_activity_fc1_k': inits.Constant(load(b,'motif_activity_fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)
    #out_dict = {'motif_activity_fc2_b': inits.Constant(load(b,'motif_activity_fc2/bias/.ATTRIBUTES/VARIABLE_VALUE')),
    #            'motif_activity_fc2_k': inits.Constant(load(b,'motif_activity_fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE'))}
    #initializers_dict.update(out_dict)

    out_dict = {'final_dense_b': inits.Constant(load(b,'target.output_target.bias')),
                'final_dense_k': inits.Constant(load(b,'target.output_target.kernel'))}
    initializers_dict.update(out_dict)
    ## load in convolutional weights
    for i in range(6):
        var_name_stem = 'target.EnformerConvolutions_0.Conv_' 
        bn_name_stem = 'target.EnformerConvolutions_0.BatchNorm_'
        bn_mut_stem = 'flax_mutables.batch_stats.EnformerConvolutions_0.BatchNorm_' #2.var

        conv1_k = var_name_stem + str(i+2) + '.kernel'
        conv1_b = var_name_stem + str(i+2) + '.bias'
        BN1_g = bn_name_stem + str(i+1) + '.scale'
        BN1_b = bn_name_stem + str(i+1) + '.bias'
        BN1_m = bn_mut_stem + str(i+1) + '.mean'
        BN1_v = bn_mut_stem + str(i+1) + '.var'

        out_dict = {'conv1_k_' + str(i): inits.Constant(load(b,conv1_k)),
                    'conv1_b_' + str(i): inits.Constant(load(b,conv1_b)),
                    'BN1_g_' + str(i): inits.Constant(load(b,BN1_g)),
                    'BN1_b_' + str(i): inits.Constant(load(b,BN1_b)),
                    'BN1_m_' + str(i): inits.Constant(load(b,BN1_m)),
                    'BN1_v_' + str(i): inits.Constant(load(b,BN1_v))}

        initializers_dict.update(out_dict)

    ## load in convolutional weights ATAC
    for i in range(2):
        var_name_stem = 'target.EnformerConvolutions_1.Conv_' 
        bn_name_stem = 'target.EnformerConvolutions_1.BatchNorm_'
        bn_mut_stem = 'flax_mutables.batch_stats.EnformerConvolutions_1.BatchNorm_' #2.var

        conv1_k = var_name_stem + str(i+2) + '.kernel'
        conv1_b = var_name_stem + str(i+2) + '.bias'
        BN1_g = bn_name_stem + str(i+1) + '.scale'
        BN1_b = bn_name_stem + str(i+1) + '.bias'
        BN1_m = bn_mut_stem + str(i+1) + '.mean'
        BN1_v = bn_mut_stem + str(i+1) + '.var'

        out_dict = {'conv_at1_k_' + str(i): inits.Constant(load(b,conv1_k)),
                    'conv_at1_b_' + str(i): inits.Constant(load(b,conv1_b)),
                    'BN_at1_g_' + str(i): inits.Constant(load(b,BN1_g)),
                    'BN_at1_b_' + str(i): inits.Constant(load(b,BN1_b)),
                    'BN_at1_m_' + str(i): inits.Constant(load(b,BN1_m)),
                    'BN_at1_v_' + str(i): inits.Constant(load(b,BN1_v))}

        initializers_dict.update(out_dict)

    #initializers_dict['performer_encoder_LN_b'] = inits.Constant(load(b,"performer/layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"))
    initializers_dict['performer_encoder_LN_g'] = inits.Constant(load(b,"target.Encoder_0.encoder_norm.scale"))

    for i in range(num_transformer_layers):
        var_name_stem = 'target.Encoder_0.layers_' + str(i)

        #LN_b=var_name_stem + 'layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        LN_g=var_name_stem + '.pre_attention_layer_norm.scale'
        out_dict = {'LN_g' + str(i): inits.Constant(load(b,LN_g))}
        initializers_dict.update(out_dict)

        SA_qkv_k=var_name_stem + ".attention.qkv_fused.kernel"
        SA_qkv_b=var_name_stem + ".attention.qkv_fused.bias"
        SA_O_k=var_name_stem + ".attention.out.kernel"
        SA_O_b=var_name_stem + ".attention.out.bias"

        FFN_narr_k=var_name_stem + ".mlp.wo.kernel"
        FFN_narr_b=var_name_stem + ".mlp.wo.bias"
        FFN_wide_k=var_name_stem + ".mlp.wi.kernel"
        FFN_wide_b=var_name_stem + ".mlp.wi.bias"
        FFN_LN_g=var_name_stem + ".pre_mlp_layer_norm.scale"
        
        qkv_bias = load(b,SA_qkv_b)
        qkv_bias =np.reshape(qkv_bias,
                             [3,load(b,SA_qkv_k).shape[-2],
                              load(b,SA_qkv_k).shape[-1]])
        out_dict = {'SA_k_k' + str(i): inits.Constant(load(b,SA_qkv_k)[:,1,:,:]),
                    'SA_q_k' + str(i): inits.Constant(load(b,SA_qkv_k)[:,0,:,:]),
                    'SA_v_k' + str(i): inits.Constant(load(b,SA_qkv_k)[:,2,:,:]),
                    'SA_k_b' + str(i): inits.Constant(qkv_bias[1,:,:]),
                    'SA_q_b' + str(i): inits.Constant(qkv_bias[0,:,:]),
                    'SA_v_b' + str(i): inits.Constant(qkv_bias[2,:,:]),
                    'SA_O_k' + str(i): inits.Constant(load(b,SA_O_k)),
                    'SA_O_b' + str(i): inits.Constant(load(b,SA_O_b)),
                    'FFN_narr_k' + str(i): inits.Constant(load(b,FFN_narr_k)),
                    'FFN_narr_b' + str(i): inits.Constant(load(b,FFN_narr_b)),
                    'FFN_wide_k' + str(i): inits.Constant(load(b,FFN_wide_k)),
                    'FFN_wide_b' + str(i): inits.Constant(load(b,FFN_wide_b)),
                    'FFN_LN_g' + str(i): inits.Constant(load(b,FFN_LN_g))}

        initializers_dict.update(out_dict)

    return initializers_dict