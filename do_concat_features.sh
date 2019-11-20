rootpath=$HOME/VisualSearch
overwrite=0

#featnames=pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 collection featnames [rootpath]"
    exit
fi

if [ "$#" -gt 2 ]; then
    rootpath=$3
fi


collection=$1
featnames=$2

python concat_features.py $collection $featnames --rootpath $rootpath --overwrite $overwrite

