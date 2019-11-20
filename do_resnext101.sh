source common.ini

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 test_collection [rootpath]"
    exit
fi

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

test_collection=$1

model_prefix=mxmodels80/resnext-101_rbps13k_step_3_aug_1_dist_3x2/resnext-101-1-0040
raw_feat_name=pyresnext-101_rbps13k,flatten0_output


./do_deep_feat.sh ${gpu_id} ${rootpath} ${oversample} ${overwrite} ${raw_feat_name} ${test_collection} ${model_prefix}

