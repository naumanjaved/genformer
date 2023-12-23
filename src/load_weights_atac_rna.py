import tensorflow as tf
from tensorflow.keras import mixed_precision

def get_initializers_genformer_ft(checkpoint_path,
                                  num_transformer_layersl):


    inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    reader = tf.train.load_checkpoint(checkpoint_path)

    initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('stem_conv/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_b': inits.Constant(reader.get_tensor('stem_conv/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_k': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}

    out_dict = {'stem_conv_atac_k': inits.Constant(reader.get_tensor('stem_conv_atac/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_atac_b': inits.Constant(reader.get_tensor('stem_conv_atac/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_k': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_g': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_m': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_v': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)

    out_dict = {'final_point_k': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_g': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_m': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_v': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)

    out_dict = {'motif_activity_fc1_b': inits.Constant(reader.get_tensor('motif_activity_fc1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                'motif_activity_fc1_k': inits.Constant(reader.get_tensor('motif_activity_fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)
    out_dict = {'motif_activity_fc2_b': inits.Constant(reader.get_tensor('motif_activity_fc2/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                'motif_activity_fc2_k': inits.Constant(reader.get_tensor('motif_activity_fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)

    out_dict = {'final_dense_b': inits.Constant(reader.get_tensor('final_dense_profile/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                'final_dense_k': inits.Constant(reader.get_tensor('final_dense_profile/kernel/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)
    ## load in convolutional weights

    for i in range(6):
        var_name_stem = 'conv_tower/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

        out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                    'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v))}

        initializers_dict.update(out_dict)

    ## load in convolutional weights ATAC
    for i in range(2):
        var_name_stem = 'conv_tower_atac/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

        out_dict = {'conv_at1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv_at1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN_at1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN_at1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN_at1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                    'BN_at1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v))}

        initializers_dict.update(out_dict)

    initializers_dict['performer_encoder_LN_b'] = inits.Constant(reader.get_tensor("performer/layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"))
    initializers_dict['performer_encoder_LN_g'] = inits.Constant(reader.get_tensor("performer/layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"))

    for i in range(num_transformer_layers):
        var_name_stem = 'performer/layers/' + str(i) + '/' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        LN_b=var_name_stem + 'layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        LN_g=var_name_stem + 'layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        out_dict = {'LN_b' + str(i): inits.Constant(reader.get_tensor(LN_b)),
                    'LN_g' + str(i): inits.Constant(reader.get_tensor(LN_g))}
        initializers_dict.update(out_dict)

        SA_k=var_name_stem + "self_attention/key_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_q=var_name_stem + "self_attention/query_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_v=var_name_stem + "self_attention/value_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_O=var_name_stem + "self_attention/output_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"

        FFN_narr_k=var_name_stem + "FFN/FFN_dense_narrow/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_narr_b=var_name_stem + "FFN/FFN_dense_narrow/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_k=var_name_stem + "FFN/FFN_dense_wide/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_b=var_name_stem + "FFN/FFN_dense_wide/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_b=var_name_stem + "FFN/FFN_layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_g=var_name_stem + "FFN/FFN_layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"


        out_dict = {'SA_k' + str(i): inits.Constant(reader.get_tensor(SA_k)),
                    'SA_q' + str(i): inits.Constant(reader.get_tensor(SA_q)),
                    'SA_v' + str(i): inits.Constant(reader.get_tensor(SA_v)),
                    'SA_O' + str(i): inits.Constant(reader.get_tensor(SA_O)),
                    'FFN_narr_k' + str(i): inits.Constant(reader.get_tensor(FFN_narr_k)),
                    'FFN_narr_b' + str(i): inits.Constant(reader.get_tensor(FFN_narr_b)),
                    'FFN_wide_k' + str(i): inits.Constant(reader.get_tensor(FFN_wide_k)),
                    'FFN_wide_b' + str(i): inits.Constant(reader.get_tensor(FFN_wide_b)),
                    'FFN_LN_b' + str(i): inits.Constant(reader.get_tensor(FFN_LN_b)),
                    'FFN_LN_g' + str(i): inits.Constant(reader.get_tensor(FFN_LN_g))}

        initializers_dict.update(out_dict)

    return initializers_dict
