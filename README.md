# dsfd

## introduction

A tensorflow implement dsfd, and there is something different with the origin paper.

It‘s a ssd-like object detect framework, but slightly different, combines lots of tricks for face detection, such as dual-shot, dense anchor match, FPN and so on.

now it is mainly optimised about face detection, and borrows tons of code from tensorpack

it achieves 0.982 on FDDB, not tested on widerface. 


ps, the code maybe not that clear, please be patience, and i am still working on it, and forgive me for my poor english :)

![FDDB eval](https://github.com/610265158/dsfd_tensorflow/blob/master/Figure_1.png)

## requirment

tensorflow1.12

tensorpack for data provider

opencv

python 3.6

### anchor

if u like to show the anchor stratergy, u could simply run :

`python anchor/utils.py`


it will draw the anchor one by one,


### data augmentation

if u like to know how the data augmentation works, run :

`python data/augmentor/augmentation.py`


### data formate and prepare data

u should prepare the data like this:

...../9_Press_Conference_Press_Conference_9_659.jpg| 483,195,735,543,1

one line for one pic

**caution! class should start from 1, 0 means bg**


download widerface data from http://shuoyang1213.me/WIDERFACE/

and release the WIDER_train, WIDER_val and wider_face_split into ./WIDER, or somewhere u like,then run

`python prepare_wider_data.py` it will produce train.txt and val.txt


### train
1.download the imagenet pretrained resnet50 model from http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

release it in the root dir,
as in train_config.py set **config.MODEL.pretrained_model='resnet_v1_50.ckpt',config.MODEL.continue_train=False**

2.but if u want to train from scratch set config.MODEL.pretrained_model=None,

3.if recover from a complet pretrained model  set **config.MODEL.pretrained_model='yourmodel.ckpt',config.MODEL.continue_train=True**

THEN, run:

`python train.py`

and if u want to check the data when training, u could set vis in train_config.py as True


#### ** CAUTION， WHEN USE TENSORPACK FOR DATA PROVIDER, some change is needed. **
#### in lib/python3.6/site-packages/tensorpack/dataflow line 71-96 , to make the iterator unstoppable, change as below
```
 71 class DataFromList(RNGDataFlow):
 72     """ Wrap a list of datapoints to a DataFlow"""
 73 
 74     def __init__(self, lst, shuffle=True):
 75         """
 76         Args:
 77             lst (list): input list. Each element is a datapoint.
 78             shuffle (bool): shuffle data.
 79         """
 80         super(DataFromList, self).__init__()
 81         self.lst = lst
 82         self.shuffle = shuffle
 83     
 84     #def __len__(self):
 85     #    return len(self.lst)
 86 
 87     def __iter__(self):
 88         if not self.shuffle:
 89             for k in self.lst:
 90                 yield k
 91         else:
 92             while True:
 93                 idxs = np.arange(len(self.lst))
 94                 self.rng.shuffle(idxs)
 95                 for k in idxs:
 96                     yield self.lst[k]
```




### visualization
![A demo](https://github.com/610265158/dsfd_tensorflow/blob/master/res_screenshot_11.05.2019.png)

(caution: i dont know where the demo picture coms from, if u think it's a tort, i would like to delet it)

`python vis.py`

u can check th code in vis.py to make it runable, it's simple.


download a pretrained model(detector.pb) from https://pan.baidu.com/s/1SRMoJIcqHRoVydl2XIZ3lA (code m3bg)
put it in to './model/detector.pb'

