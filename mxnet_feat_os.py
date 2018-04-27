'''
Weiyu Lan, Xirong Li, Jianfeng Dong, Fluency-Guided Cross-Lingual Image Captioning, ACM MM 2017
'''

from __future__ import print_function, division
import mxnet as mx
import os,sys
import logging
import traceback
import numpy as np
from collections import namedtuple

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

from constant import *

IMG_SIZE = 256
CROP_SIZE = 224
ZERO_IMAGE = np.zeros((IMG_SIZE, IMG_SIZE, 3))
INVALID_ID = 'INVALID'
DEFAULT_OVERSAMPLE = 1
DEVICE_ID = 0

Batch = namedtuple('Batch', ['data'])


def img_oversample(raw_img, width=IMG_SIZE, height=IMG_SIZE, crop_dims=CROP_SIZE):
    cropped_image, _ = mx.image.center_crop(raw_img, (crop_dims, crop_dims))
    cropped_image_1 = mx.image.fixed_crop(raw_img, 0, 0, crop_dims, crop_dims)
    cropped_image_2 = mx.image.fixed_crop(raw_img, 0, height-crop_dims, crop_dims, crop_dims)
    cropped_image_3 = mx.image.fixed_crop(raw_img, width-crop_dims, 0, crop_dims, crop_dims)
    cropped_image_4 = mx.image.fixed_crop(raw_img, width-crop_dims, height-crop_dims, crop_dims, crop_dims)
    img_list = [cropped_image.asnumpy(), cropped_image_1.asnumpy(), cropped_image_2.asnumpy(), 
                cropped_image_3.asnumpy(), cropped_image_4.asnumpy()]
    return img_list

def preprocess_images(inputs, width=IMG_SIZE, height=IMG_SIZE, crop_dims=CROP_SIZE, oversample=True):
    # Scale to standardize input dimensions.
    input_ = []

    for ix, in_ in enumerate(inputs):
        raw_img = mx.image.imresize(in_, width, height)
        if oversample:
            # Generate center, corner, and mirrored crops.
            input_.extend(img_oversample(raw_img, width, height, crop_dims))
            input_.extend(img_oversample(mx.nd.flip(raw_img, axis=1), width, height, crop_dims))
        else:
            cropped_image, _ = mx.image.center_crop(raw_img, (crop_dims, crop_dims))
            input_.append(cropped_image.asnumpy())

    input_ = mx.nd.array(input_)
    input_ = mx.nd.swapaxes(input_, 1, 3)
    input_ = mx.nd.swapaxes(input_, 2, 3)
    return Batch([input_])


def get_feat_extractor(model_prefix, epoch=DEFAULT_EPOCH, gpuid=DEVICE_ID, batch_size=1, oversample=True):
    layer = 'flatten0_output'
    batch_size = batch_size*10 if oversample else batch_size
    print (batch_size, model_prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    all_layers = sym.get_internals()
    fe_sym = all_layers[layer]
    if gpuid >= 0:
        fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(gpuid), label_names=None)
    else:
        fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
    fe_mod.bind(for_training=False, data_shapes=[('data', (batch_size,3,CROP_SIZE,CROP_SIZE))])
    fe_mod.set_params(arg_params, aux_params)
    return fe_mod


def extract_feature(model, batch_size, imset, path_imgs, oversample=True):
    assert(len(imset)==1)
    impath = path_imgs[0]
    img = mx.image.imdecode(open(impath).read())

    mxnet_in = preprocess_images([img], oversample=oversample)
    model.forward(mxnet_in)
    features = model.get_outputs()[0].asnumpy()
    if oversample:
        features = features.reshape((len(features)//10, 10,-1)).mean(1)
    return (imset, features)


if __name__ == '__main__':
    from constant import *

    oversample = False
    model_prefix = os.path.join(ROOT_PATH, DEFAULT_MODEL_PREFIX)
    model = get_feat_extractor(model_prefix, gpuid=-1, oversample=oversample)
    imset = str.split('COCO_train2014_000000042196')
    path_imgs = ['%s.jpg'%x for x in imset]
    _, features = extract_feature(model, 1, imset, path_imgs, oversample=oversample)
    print (features.shape)
 
