rootpath=$HOME/VisualSearch

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 data_type [rootpath]"
    exit
fi

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

if [ $1 = '1k' ]; then
    mkdir -p ${rootpath}/mxnet_models/imagenet-1k/resnet-152
    cd ${rootpath}/mxnet_models/imagenet-1k/resnet-152

    wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-symbol.json
    wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-0000.params

elif [ $1 = '11k' ]; then
    mkdir -p ${rootpath}/mxnet_models/imagenet-11k/resnet-152
    cd ${rootpath}/mxnet_models/imagenet-11k/resnet-152

    wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json
    wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params
elif [ $1 = 'v3' ]; then
    mkdir -p $HOME/premodels/yt8m
    cd $HOME/premodels/yt8m

    wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    wget http://data.yt8m.org/yt8m_pca.tgz

    tar zxvf inception-2015-12-05.tgz
    tar zxvf yt8m_pca.tgz

else
    echo "data_type is invalid"
fi
