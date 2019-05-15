#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu

config.TRAIN = edict()
config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 4                    ###A big batch size may achieve a better result the the memory is a problem
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 2000
config.TRAIN.train_set_size=16000  ###########u need be sure
config.TRAIN.val_set_size=2800
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.00001,0.0001,0.001,0.0001,0.00001,0.000001]
config.TRAIN.lr_decay_every_step = [8000,12000,60000,80000,100000]

config.TRAIN.weight_decay_factor = 5.e-4
config.TRAIN.dropout=0.5  ##no use
config.TRAIN.vis=False

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False    ##'GN' OR 'BN'
config.TEST = edict()
config.TEST.PARALLEL_ITERATIONS=8
# Smaller threshold value gives significantly better mAP. But we use 0.05 for consistency with Detectron.
# mAP with 1e-4 threshold can be found at https://github.com/tensorpack/tensorpack/commit/26321ae58120af2568bdbf2269f32aa708d425a8#diff-61085c48abee915b584027e1085e1043  # noqa
config.TEST.RESULT_SCORE_THRESH = 0.05



config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.NUM_CATEGORY=1  ###face 1  voc 20 coco 80
config.DATA.NUM_CLASS = config.DATA.NUM_CATEGORY + 1  # +1 background

config.DATA.PIXEL_MEAN = [123.675, 116.28, 103.53]   ###rgb
config.DATA.PIXEL_STD = [58.395, 57.12, 57.375]

config.DATA.hin = 640  # input size
config.DATA.win= 640
config.DATA.MAX_SIZE=640
config.DATA.cover_small_face=5        #one of the
####ssd generally not suppport muti scale
config.DATA.MUTISCALE=False                #if muti scale set False  then config.DATA.hin will be the inputsize
config.DATA.scales=(960,800,640,512,480,416,320)


# anchors -------------------------
config.ANCHOR = edict()
config.ANCHOR.rect=True
config.ANCHOR.rect_longer=True
config.ANCHOR.ANCHOR_STRIDE = 16
config.ANCHOR.ANCHOR_SIZES = (16,32,64, 128, 256, 512)   # sqrtarea of the anchor box
config.ANCHOR.ANCHOR_STRIDES = (4, 8,16, 32, 64, 128)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.ANCHOR.ANCHOR_RATIOS = (1., 4.) ######
config.ANCHOR.POSITIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.NEGATIVE_ANCHOR_THRESH = 0.3
config.ANCHOR.AVG_MATCHES=20
config.ANCHOR.super_match=True

config.ANCHOR.MAX_ANCHORS_MATCHES = 2048  ##nouse

from anchor.utils import get_all_anchors_fpn

anchors_per_level = get_all_anchors_fpn()
flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)
config.ANCHOR.achors=all_anchors_flatten
config.ANCHOR.achors_num=len(all_anchors_flatten)

print(len(all_anchors_flatten))

# basemodel ----------------------
config.MODEL = edict()
config.MODEL.continue_train=False ### revover from a trained model
config.MODEL.model_path = './model/'  # save directory
config.MODEL.net_structure='resnet_v1_50' ######'resnet_v1_50,resnet_v1_101,mobilenet
config.MODEL.pretrained_model='resnet_v1_50.ckpt'

###resnet_v1_50 as basemodel
# config.MODEL = edict()
# config.MODEL.continue_train=False ### revover from a trained model
# config.MODEL.model_path = './model/'  # save directory
# config.MODEL.net_structure='resnet_v1_101' ######'resnet_v1_50,resnet_v1_101,mobilenet
# config.MODEL.pretrained_model='resnet_v1_101.ckpt'

##mobilenet as basemodel,but mobilenet is failed, keedchecking ingingining
# config.MODEL = edict()
# config.MODEL.continue_train=False ### revover from a trained model
# config.MODEL.model_path = './model/'  # save directory
# config.MODEL.net_structure='MobilenetV1' ######'resnet_v1_50,resnet_v1_101,mobilenet
# config.MODEL.pretrained_model='mobilenet_v1_0.5_160.ckpt'

config.MODEL.fpn=False
config.MODEL.dual_mode=True
config.MODEL.maxout=True
config.MODEL.context=True
config.MODEL.ohem=True
config.MODEL.focal_loss=True

config.MODEL.loss_to_use="classification"
config.MODEL.loc_loss_weight= 0.0
config.MODEL.cls_loss_weight= 1.0
config.MODEL.num_hard_examples= 500
config.MODEL.nms_threshold= 0.99
config.MODEL.max_negatives_per_positive= 3.0
config.MODEL.min_negatives_per_image=30
