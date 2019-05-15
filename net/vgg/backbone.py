import tensorflow as tf
import tensorflow.contrib.slim as slim


from net.vgg.vgg import vgg_16


from net.lightnet.magic import block
from net.resnet.basemodel import resnet50, resnet_arg_scope

from net.lightnet.magic import block



def extra_feature(x):

    net = block(x[-1], num_units=2, out_channels=256, scope='extra_Stage1')
    x.append(net)
    net = block(net, num_units=2, out_channels=256, scope='extra_Stage2')
    x.append(net)
    return x



def vgg_ssd(image,L2_reg,is_training=True):

    fms_vgg=vgg_16(image)
    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training, bn_trainable=True,
                                         data_format='NHWC')):
        fms_6=extra_feature(fms_vgg)


    return fms_6
