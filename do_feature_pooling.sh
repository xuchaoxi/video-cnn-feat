source common.ini

#feature=pyresnet-152_imagenet11k,flatten0_output,os
pooling=mean

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 collection feat_name [rootpath]"
    exit
fi

if [ "$#" -gt 2 ]; then
    rootpath=$3
fi

collection=$1
feature=$2


python videocnn/feature_pooling.py $collection --overwrite $overwrite --rootpath $rootpath --feature $feature --pooling $pooling
