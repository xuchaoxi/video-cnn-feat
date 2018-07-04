gpu_id=0  # set to -1 if gpu is not available
rootpath=$HOME/VisualSearch
oversample=0
overwrite=1

raw_feat_name=pyinception-v3,pool_3_reshape

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 test_collection [rootpath]"
    exit
fi

test_collection=$1

model_prefix=inception-v3
mxmodel_dir=$HOME/premodels/yt8m

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

./do_deep_feat.sh ${gpu_id} ${rootpath} ${oversample} ${overwrite} ${raw_feat_name} ${test_collection} ${model_prefix} ${mxmodel_dir}

