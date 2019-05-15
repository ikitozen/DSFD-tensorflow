import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial

from train_config import config as cfg

from net.resnet.basemodel import resnet50, resnet101,resnet_arg_scope

from net.lightnet.magic import block


resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=True)



def resnet_ssd(image,L2_reg,is_training=True,data_format='NHWC'):


    resnet=resnet101 if '101' in cfg.MODEL.net_structure else resnet50

    if cfg.TRAIN.lock_basenet_bn:
        resnet_fms = resnet(image, L2_reg,bn_is_training=False, bn_trainable=True,data_format=data_format)
    else:
        resnet_fms = resnet(image, L2_reg, bn_is_training=is_training, bn_trainable=True, data_format=data_format)
    print('resnet50 backbone output:',resnet_fms)


    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training, bn_trainable=True,
                                         data_format=data_format)):

        # net = block(resnet_fms[-1], num_units=2, out_channels=512, scope='extra_Stage1')
        # resnet_fms.append(net)
        # net = block(net, num_units=2, out_channels=512, scope='extra_Stage2')
        # resnet_fms.append(net)
        net = slim.conv2d(resnet_fms[-1], 512, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_1_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_1_2')
        resnet_fms.append(net)
        net = slim.conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='extra_conv_2_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='extra_conv_2_2')
        resnet_fms.append(net)
        print('extra resnet50 backbone output:', resnet_fms)

    return resnet_fms
