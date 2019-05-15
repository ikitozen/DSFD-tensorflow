import tensorflow as tf

import tensorflow.contrib.slim as slim
from train_config import config as cfg



from net.simplenet.simple_nn import shufflenet_arg_scope

from net.simplenet.simple_nn import simple_nn

def halo(x,scope):
    in_channels = x.shape[3].value
    with tf.variable_scope(scope):
        with tf.variable_scope('first_branch'):
            x1 = slim.conv2d(x, 128, [3, 3], stride=1,rate=2, activation_fn=None, scope='_conv_1_1')
        with tf.variable_scope('second_branch'):
            x2 = slim.conv2d(x, 128, [3, 3], stride=1,rate=2, activation_fn=tf.nn.relu, scope='_conv_2_1')
            x2 = slim.conv2d(x2, 128, [3, 3], stride=1,rate=2, activation_fn=None, scope='_conv_2_2')
    x = tf.concat([x1, x2], axis=3)
    return x
def cpm(product,dim,scope):

    with tf.variable_scope(scope):

        eyes_1=slim.conv2d(product, dim//2, [3, 3], stride=1,rate=1, activation_fn=tf.nn.relu, scope='eyes_1')

        eyes_2_1=slim.conv2d(product, dim//2, [3, 3], stride=1,rate=2,  activation_fn=tf.nn.relu, scope='eyes_2_1')
        eyes_2=slim.conv2d(eyes_2_1, dim//4, [3, 3], stride=1,rate=1,  activation_fn=tf.nn.relu, scope='eyes_2')

        eyes_3_1 = slim.conv2d(eyes_2_1, dim//2, [3, 3], stride=1, rate=2, activation_fn=tf.nn.relu, scope='eyes_3_1')
        eyes_3 = slim.conv2d(eyes_3_1, dim//4, [3, 3], stride=1,rate=1,  activation_fn=tf.nn.relu, scope='eyes_3')

        fme_res = tf.concat([eyes_1, eyes_2,eyes_3], axis=3)

    return fme_res


if 'MobilenetV1' in cfg.MODEL.net_structure:
    resnet_dims=[64,128,256,512,512,256]
elif 'resnet' in cfg.MODEL.net_structure:
    resnet_dims=[256,512,1024,2048,512,256]
else:
    ssd_backbne = None



def create_fem_net(blocks, L2_reg,is_training, trainable=True,data_format='NHWC'):

    global_fms = []

    last_fm = None
    initializer = tf.contrib.layers.xavier_initializer()
    for i, block in enumerate(reversed(blocks)):
        with slim.arg_scope(shufflenet_arg_scope(weight_decay=L2_reg,is_training=is_training)):

            dim = resnet_dims[6-i-1]

            if i>=3:
                print(block)
                print(last_fm)
                lateral = slim.conv2d(block, dim, [1, 1],
                    trainable=trainable, weights_initializer=initializer,
                    padding='SAME', activation_fn=None,
                    scope='lateral/res{}'.format(5-i))
            else:
                lateral=block
            if last_fm is not None and i >=3:
                upsample = slim.conv2d(last_fm, dim, [1, 1],
                                       trainable=trainable, weights_initializer=initializer,
                                       padding='SAME', activation_fn=None,
                                       scope='merge/res{}'.format(5 - i), data_format=data_format)
                upsample = tf.keras.layers.UpSampling2D(data_format='channels_last' if data_format=='NHWC' else 'channels_first')(upsample)

                last_fm = lateral* upsample

            else:
                last_fm = lateral

        global_fms.append(last_fm)

    global_fms.reverse()

    global_fems_fms=[]
    with slim.arg_scope(shufflenet_arg_scope(weight_decay=L2_reg, is_training=is_training)):
        for i, fem in enumerate(global_fms):
            tmp_res=cpm(fem,dim=resnet_dims[i],scope='fems%d'%i)
            global_fems_fms.append(tmp_res)

    return global_fems_fms
