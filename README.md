# Extracting CNN features from video frames by MXNet

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
# Download resnet-152 model pre-trained on imagenet-11k
./do_prepare.sh
```

## Get started

Feature extraction is performed in the following four steps. Skip the first step if frames are already there.

### Step 1. Extract frames from videos 

Our code assumes the following data organization. We provide the `toydata` folder as an example.
```
collection_name
+ VideoData
+ ImageData
+ id.imagepath.txt
```
Video files are stored in the `VideoData` folder. Frame files are in the `ImageData`folder. 
+ Video filenames shall end with `.mp4`, `.avi`, `.webm`, or `.gif`.
+ Frame filenames shall end with `.jpg`.


```
collection=toydata
./do_extract_frames.sh $collection
```

### Step 2. Extract frame-level CNN features

```
./do_resnet152-11k.sh $collection
./do_resnext101.sh $collection
```

### Step 3. Obtain video-level CNN features
```
./do_feature_pooling $collection pyresnet-152_imagenet11k,flatten0_output,os
./do_feature_pooling $collection pyresnext-101_rbps13k,flatten0_output,os
```

### Step 4. Feature concatenation
```
featname=pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os
./do_concat_features.sh $collection $featname
```

# References

