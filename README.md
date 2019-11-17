# Deep CNN Feature by MxNet

## Requirements

### Environments

* Ubuntu 16.04
* CUDA 9.0
* python 2.7
* opencv-python
* mxnet-cu90 
* numpy

We used virtualenv to setup a deep learning workspace that supports MXNet. Run the following script to install the required packages.
```
virtualenv --system-site-packages ~/cnn_feat
source ~/cnn_feat/bin/activate
pip install -r requirements.txt
deactivate
```

### Required models

Run `do_prepare.sh` to download pre-trained CNN models.

```
# Download resnet-152 model pre-trained on imagenet-1k
./do_prepare.sh 1k

# Download resnet-152 model pre-trained on imagenet-11k
./do_prepare.sh 11k

# Download inception-v3 model pre-trained on yt8m
./do_prepare.sh v3
```

## Get started

### Data

Store videos into the `VideoData` under collection folder and images into `ImageData` if you have extracted frames from videos.

### Extract frames from videos

```
source ~/cnn_feat/bin/activate
cd videocnn
python generate_videopath.py $collection_name
python video2frames.py $collection_name
```

### Extract CNN features

```
source ~/cnn_feat/bin/activate
./do_resnet152-11k.sh $collection_name
./do_resnet152-1k.sh $collection_name
./do_resnext101.sh $collection_name
```
