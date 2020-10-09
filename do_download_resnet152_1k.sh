source common.ini

mkdir -p ${rootpath}/mxnet_models/imagenet-1k/resnet-152
cd ${rootpath}/mxnet_models/imagenet-1k/resnet-152

wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-symbol.json
wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-0000.params
