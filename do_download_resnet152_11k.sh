source common.ini

mkdir -p ${rootpath}/mxnet_models/imagenet-11k/resnet-152
cd ${rootpath}/mxnet_models/imagenet-11k/resnet-152

wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params
