gpu_id=0  # set to -1 if gpu is not available
rootpath=$HOME/VisualSearch
oversample=1
overwrite=1

raw_feat_name=pyresnext-101_rbps13k,flatten0_output,crop

imgpath_file=id.imagepath.ImageData_crop.txt

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 test_collection [rootpath]"
    exit
fi

test_collection=$1

model_prefix=mxmodels80/resnext-101_rbps13k_step_3_aug_1_dist_3x2/resnext-101-1
mxmodel_dir=$rootpath/mxmodels80/resnext-101_rbps13k_step_3_aug_1_dist_3x2

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

./do_deep_feat.sh ${gpu_id} ${rootpath} ${oversample} ${overwrite} ${raw_feat_name} ${test_collection} ${model_prefix} ${mxmodel_dir} ${imgpath_file}

