source common.ini
overwrite=0

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 collection [rootpath]"
    exit
fi

if [ "$#" -gt 1 ]; then
    rootpath=$2
fi

collection=$1

python videocnn/generate_videopath.py $collection --rootpath $rootpath --overwrite $overwrite

python videocnn/video2frames.py $collection --rootpath $rootpath --overwrite $overwrite

