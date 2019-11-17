gpu_id=1  # set to -1 if gpu is not available
rootpath=$HOME/VisualSearch
oversample=1
overwrite=0

raw_feat_name=pyresnet-152_imagenet11k,flatten0_output
imgpath_file=id.imagepath.txt

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 test_collection [rootpath]"
    exit
fi

test_collection=$1

model_prefix=mxnet_models/imagenet-11k/resnet-152/resnet-152
mxmodel_dir=$rootpath/mxnet_models/imagenet-11k/resnet-152

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

./do_deep_feat.sh ${gpu_id} ${rootpath} ${oversample} ${overwrite} ${raw_feat_name} ${test_collection} ${model_prefix} ${mxmodel_dir} ${imgpath_file}

