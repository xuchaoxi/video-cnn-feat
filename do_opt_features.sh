rootpath=$HOME/VisualSearch
set_style=ImageSets
overwrite=1

#featname=pyresnext-101_rbps13k,flatten0_output,os
featname=pyinception-v3,pool_3_reshape

collection=tgif
sub_collections=tgif-tmp@tgifval

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
    for collect in tv2016test1 tv2016test2
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

