source common.ini

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 test_collection [rootpath]"
    exit
fi

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

test_collection=$1

model_prefix=mxnet_models/imagenet-11k/resnet-152/resnet-152-0000
raw_feat_name=pyresnet-152_imagenet11k,flatten0_output

./do_deep_feat.sh ${gpu_id} ${rootpath} ${oversample} ${overwrite} ${raw_feat_name} ${test_collection} ${model_prefix}

