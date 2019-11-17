rootpath=$HOME/VisualSearch/activitynet
set_style=ImageSets
set_style=VideoSets
overwrite=0

featname=pyinception-v3,pool_3_reshape
featname=pyresnext-101_rbps13k,flatten0_output,os
featname=mean_pyresnext-101_rbps13k,flatten0_output,os
featname=pyresnet-152_imagenet11k,flatten0_output,os

collection=tgif-msrvtt10k-activitynet_vdc
sub_collections=tgif-msrvtt10k@activitynet_vdc
collection=activitynet
sub_collections=data_0@data_1@data_2@data_3@data_4@data_5@data_6@data_7@data_8@data_9@data_10@data_11

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 python_script"
    exit
fi

BASEDIR=$(dirname "$0")

python ${BASEDIR}/$1 $collection $featname ${sub_collections} ${set_style} \
    --overwrite $overwrite \
    --rootpath $rootpath

#IFS='@' read -r -a array <<< "${sub_collections}"

if [ $1 = split* ]; then
    for collect in v3c1_sub
    do
        feat_dir=$rootpath/$collect/FeatureData/$featname
        feat_file=${feat_dir}/id.feature.txt

        if [ -f ${feat_file} ]; then
            python ${BASEDIR}/txt2bin.py 2048 ${feat_file} 0 ${feat_dir} --overwrite 1
            rm ${feat_file}
        fi
    done

elif [ $1 = join_* ]; then
    feat_dir=$rootpath/$collection/FeatureData/$featname
    feat_file=${feat_dir}/id.feature.txt

    if [ -f ${feat_file} ]; then
        python ${BASEDIR}/txt2bin.py 2048 ${feat_file} 0 ${feat_dir} --overwrite 1
        rm ${feat_file}
    fi
fi

