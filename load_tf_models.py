import tensorflow as tf
import torch
import os


def load_embeddings(module, prefix, tf_path):
    """loads embeddings from tf in the pytorch model
    :param module: the pytorch embeddings module
    :param prefix: the prefix corresponding to module in the weights
    :param tf_path: the path to load the different weights from

    :return: the list of weights (in tf_path) that were loaded by this function
    """
    weights_name = prefix + 'lookup_table'
    tf_weights = tf.train.load_variable(tf_path, weights_name)
    module.weight.data.copy_(torch.from_numpy(tf_weights))
    return [weights_name]


def load_channelnorm(module, prefix, tf_path):
    """loads a channel norm layer from tf in the pytorch model
    :param module: the pytorch channel norm module
    :param prefix: the prefix corresponding to module in the weights
    :param tf_path: the path to load the different weights from

    :return: the list of weights (in tf_path) that were loaded by this function
    """
    weights_name = [prefix + 'beta', prefix + 'gamma']
    module.beta.data.copy_(torch.from_numpy(tf.train.load_variable(tf_path, weights_name[0]).reshape(module.shape)))
    module.gamma.data.copy_(torch.from_numpy(tf.train.load_variable(tf_path, weights_name[1]).reshape(module.shape)))
    return weights_name


def load_conv1dnormact(module, prefix, tf_path):
    """loads conv1dnormact from tf in the pytorch model
    :param module: the pytorch conv1dnormact module
    :param prefix: the prefix corresponding to module in the weights
    :param tf_path: the path to load the different weights from

    :return: the list of weights (in tf_path) that were loaded by this function
    """
    weights_name = [prefix + 'conv1d/bias', prefix + 'conv1d/kernel']
    module.conv1d.bias.data.copy_(torch.from_numpy(tf.train.load_variable(tf_path, weights_name[0])))
    conv_weights = tf.train.load_variable(tf_path, weights_name[1]).transpose((2, 1, 0))  # (kernel, in, out) ->
    # (out, in, kernel)
    module.conv1d.weight.data.copy_(torch.from_numpy(conv_weights))

    # loads channel norm
    weights_name += load_channelnorm(module.normalize, prefix + 'normalize/', tf_path)
    return weights_name


def load_convtranspose1dnormact(module, prefix, tf_path):
    """loads ConvTranspose1DNormAct from tf in the pytorch model
    :param module: the pytorch ConvTranspose1DNormAct module
    :param prefix: the prefix corresponding to module in the weights
    :param tf_path: the path to load the different weights from

    :return: the list of weights (in tf_path) that were loaded by this function
    """
    weights_name = [prefix + 'conv2d_transpose/bias', prefix + 'conv2d_transpose/kernel']
    module.convtranspose1d.bias.data.copy_(torch.from_numpy(tf.train.load_variable(tf_path, weights_name[0])))
    conv_weights = tf.train.load_variable(tf_path, weights_name[1]).squeeze().transpose((2, 1, 0))
    # (kernel_h, kernel_w, out, in) -> (in, out, kernel)
    module.convtranspose1d.weight.data.copy_(torch.from_numpy(conv_weights))

    # loads channel norm
    weights_name += load_channelnorm(module.normalize, prefix + 'normalize/', tf_path)
    return weights_name


def load_highwayconv(module, prefix, tf_path):
    """loads highway conv from tf in the pytorch model
    :param module: the pytorch highway conv module
    :param prefix: the prefix corresponding to module in the weights
    :param tf_path: the path to load the different weights from

    :return: the list of weights (in tf_path) that were loaded by this function
    """
    weights_name = [prefix + 'conv1d/bias', prefix + 'conv1d/kernel']
    module.conv1d.bias.data.copy_(torch.from_numpy(tf.train.load_variable(tf_path, weights_name[0])))
    conv_weights = tf.train.load_variable(tf_path, weights_name[1]).transpose((2, 1, 0))  # (kernel, in, out) ->
    # (out, in, kernel)
    module.conv1d.weight.data.copy_(torch.from_numpy(conv_weights))

    # loads the two channel norm
    weights_name += load_channelnorm(module.normalize_1, prefix + 'H1/', tf_path)
    weights_name += load_channelnorm(module.normalize_2, prefix + 'H2/', tf_path)
    return weights_name


def load_t2m_from_tf(model, checkpoint):
    """loads the weights from a tensorflow checkpoint of t2m into the pytorch model"""
    print("Starting to load Text2Mel")
    tf_path = os.path.abspath(checkpoint)
    var_loaded = []

    print("loading TextEnc")
    prefix = "Text2Mel/TextEnc/"
    module = model.textEnc
    var_loaded += load_embeddings(module.embed, prefix + 'embed_1/', tf_path)
    var_loaded += load_conv1dnormact(module.C_1, prefix + 'C_2/', tf_path)
    var_loaded += load_conv1dnormact(module.C_2, prefix + 'C_3/', tf_path)
    for i, hc_module in enumerate(module.HCs_1):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{4+i}/', tf_path)
    for i, hc_module in enumerate(module.HCs_2):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{12+i}/', tf_path)
    for i, hc_module in enumerate(module.HCs_3):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{14+i}/', tf_path)

    print("loading AudioEnc")
    prefix = "Text2Mel/AudioEnc/"
    module = model.audioEnc
    for layer in ["C_1", "C_2", "C_3"]:
        var_loaded += load_conv1dnormact(getattr(module, layer), prefix + layer + '/', tf_path)
    for i, hc_module in enumerate(module.HCs_1):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{4+i}/', tf_path)
    for i, hc_module in enumerate(module.HCs_2):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{12+i}/', tf_path)

    print("loading AudioDec")
    prefix = "Text2Mel/AudioDec/"
    module = model.audioDec
    var_loaded += load_conv1dnormact(module.C_1, prefix + 'C_1/', tf_path)
    for i, hc_module in enumerate(module.HCs_1):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{2+i}/', tf_path)
    for i, hc_module in enumerate(module.HCs_2):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{6 + i}/', tf_path)
    for i, conv_module in enumerate(module.Cs_2):
        var_loaded += load_conv1dnormact(conv_module, prefix + f'C_{8 + i}/', tf_path)
    var_loaded += load_conv1dnormact(module.C_3, prefix + f'C_11/', tf_path)

    var_loaded = set(var_loaded)
    var_not_loaded = [var for var, _ in tf.train.list_variables(tf_path) if
                      (var not in var_loaded and var.startswith("Text2Mel") and "Adam" not in var)]
    if len(var_not_loaded) != 0:
        print("The following variables are present in the checkpoint and belong to Text2Mel but were not loaded:")
        print(var_not_loaded)
    print("Text2Mel loaded")


def load_ssrn_from_tf(model, checkpoint):
    """loads the weights from a tensorflow checkpoint of ssrn into the pytorch model"""
    print("Starting to load SSRN")
    tf_path = os.path.abspath(checkpoint)
    prefix = "SSRN/"
    var_loaded = []

    var_loaded += load_conv1dnormact(model.C_1, prefix + 'C_1/', tf_path)
    for i, hc_module in enumerate(model.HCs_1):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{2+i}/', tf_path)
    for i, d_hc_module in enumerate(model.D_HCs_1):
        if i == 0 or i == 3:
            var_loaded += load_convtranspose1dnormact(d_hc_module, prefix + f'D_{4+i}/', tf_path)
        else:
            var_loaded += load_highwayconv(d_hc_module, prefix + f'HC_{4+i}/', tf_path)
    var_loaded += load_conv1dnormact(model.C_2, prefix + 'C_10/', tf_path)
    for i, hc_module in enumerate(model.HCs_2):
        var_loaded += load_highwayconv(hc_module, prefix + f'HC_{11+i}/', tf_path)
    var_loaded += load_conv1dnormact(model.C_3, prefix + 'C_13/', tf_path)
    for i, hc_module in enumerate(model.Cs_4):
        var_loaded += load_conv1dnormact(hc_module, prefix + f'C_{14+i}/', tf_path)
    var_loaded += load_conv1dnormact(model.C_5, prefix + 'C_16/', tf_path)

    var_loaded = set(var_loaded)
    var_not_loaded = [var for var, _ in tf.train.list_variables(tf_path) if
                      (var not in var_loaded and var.startswith("SSRN") and "Adam" not in var)]
    if len(var_not_loaded) != 0:
        print("The following variables are present in the checkpoint and belong to SSRN but were not loaded:")
        print(var_not_loaded)
    print("SSRN loaded")
