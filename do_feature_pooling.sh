overwrite=0
rootpath=$HOME/VisualSearch
feature=pyresnext-101_rbps13k,flatten0_output,os
feature=pyresnet-152_imagenet1k,flatten0_output,os
feature=pyinception-v3,pool_3_reshape
pooling=mean

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 collection [rootpath]"
    exit
fi

collection=$1

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

python -m videocnn.feature_pooling $collection \
    --overwrite $overwrite \
    --rootpath $rootpath \
    --feature $feature \
    --pooling $pooling
