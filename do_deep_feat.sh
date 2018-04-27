gpu_id=$1 
rootpath=$2
oversample=$3
overwrite=$4

raw_feat_name=$5

test_collection=$6

model_prefix=$7
mxmodel_dir=$8

if [ ! -d ${mxmodel_dir} ]; then
    echo "${mxmodel_dir} not found. CNN model not ready"
    exit
fi

if [ "$oversample" == 1 ]; then
    raw_feat_name=${raw_feat_name},os
fi  

BASEDIR=$(dirname "$0")

python ${BASEDIR}/generate_imagepath.py ${test_collection} --overwrite 0 --rootpath $rootpath
imglistfile=$rootpath/${test_collection}/id.imagepath.txt

if [ ! -f $imglistfile ]; then
    echo "$imglistfile does not exist"
    exit
fi

python ${BASEDIR}/extract_deep_feat.py ${test_collection} --model_prefix ${model_prefix} --oversample $oversample --gpu ${gpu_id} --overwrite $overwrite --rootpath $rootpath

feat_dir=$rootpath/${test_collection}/FeatureData/$raw_feat_name
feat_file=$feat_dir/id.feature.txt

if [ -f ${feat_file} ]; then
    python ${BASEDIR}/txt2bin.py 2048 $feat_file 0 $feat_dir --overwrite 1
    rm $feat_file
fi

if [ ! -d ${feat_dir} ]; then
    echo "$feat_dir does not exist"
    exit
fi

#python ${BASEDIR}/norm_feat.py $feat_dir --overwrite $overwrite

