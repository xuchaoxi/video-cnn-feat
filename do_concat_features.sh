rootpath=$HOME/VisualSearch
featnames=mean_pyresnext-101_rbps13k,flatten0_output,os+mean_pyresnet-152_imagenet11k,flatten0_output,os
overwrite=0

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 collection [rootpath]"
    exit
fi

collection=$1

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

python concat_features.py $collection $featnames \
    --rootpath $rootpath \
    --overwrite $overwrite


feat_dir=$rootpath/$collection/FeatureData/$featnames
feat_file=${feat_dir}/id.feature.txt

if [ -f ${feat_file} ]; then
    python txt2bin.py 4096 $feat_file 0 $feat_dir --overwrite 1
    rm $feat_file
fi

