import sys, os
import numpy as np
import logging

from constant import ROOT_PATH, PROGRESS, DEFAULT_FEAT, DEFAULT_POOLING

sys.path.extend([".", ".."])
from utils.bigfile import BigFile
from utils.generic_utils import Progbar

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def get_weight_vec(n):
    ref = n/2
    weights = [np.exp(-0.25*abs(x-ref)) for x in range(n)]
    Z = sum(weights)
    return [x/Z for x in weights]

def mean_pooling(feat_matrix):
    return feat_matrix.mean(axis=0)

def max_pooling(feat_matrix):
    return feat_matrix.max(axis=0)

def gauss_pooling(feat_matrix):
    nr_points, feat_dim = feat_matrix.shape
    weights = get_weight_vec(nr_points)
    return np.dot(weights, feat_matrix)

def get_pooling_func(pooling):
    if 'mean' == pooling:
        return mean_pooling
    elif 'max' == pooling:
        return max_pooling
    elif 'gauss' == pooling:
        return gauss_pooling
    raise Exception('unknown pooling strategy %s' % pooling)    

def process(options, collection):
    rootpath = options.rootpath
    feature = options.feature
    pooling = options.pooling
    overwrite = options.overwrite
    
    pooling_func = get_pooling_func(pooling)   
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feature)
    res_dir = os.path.join(rootpath, collection, 'FeatureData', '%s_%s' % (pooling, feature))
    
    if os.path.exists(res_dir):
        if overwrite:
            logger.info("%s exists. overwrite", res_dir)
        else:
            logger.info("%s exists. quit", res_dir)
            return 0

    feat_file = BigFile(feat_dir)
    video2frames = {}
    for frame_id in feat_file.names:
        video_id, frame_index = frame_id.rsplit('_',1)
        frame_index = int(frame_index)
        video2frames.setdefault(video_id,[]).append(frame_id)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    res_binary_file = os.path.join(res_dir, 'feature.bin')
    fw = open(res_binary_file, 'wb')
    videoset = []

    pbar = Progbar(len(video2frames))
    for video_id, frame_id_list in video2frames.iteritems():
        renamed, vectors = feat_file.read(frame_id_list)
        name2vec = dict(zip(renamed, vectors))
        frame_id_list.sort(key=lambda v: int(v.rsplit('_',1)[-1]))

        feat_matrix = np.zeros((len(renamed), len(vectors[0])))
        for i,frame_id in enumerate(frame_id_list):
            feat_matrix[i,:] = name2vec[frame_id]

        video_vec = pooling_func(feat_matrix)
        video_vec.astype(np.float32).tofile(fw)
        videoset.append(video_id)
        pbar.add(1)
    fw.close()

    fw = open(os.path.join(res_dir, 'id.txt'), 'w')
    fw.write(' '.join(videoset))
    fw.close()

    fw = open(os.path.join(res_dir,'shape.txt'), 'w')
    fw.write('%d %d' % (len(videoset), len(video_vec)))
    fw.close() 

    logger.info("%s pooling -> %dx%d video feature file", pooling, len(videoset), len(video_vec))

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--feature", default=DEFAULT_FEAT, type="string", help="cnn feature (default: %s)" % DEFAULT_FEAT)
    parser.add_option("--pooling", default=DEFAULT_POOLING, type="string", help="pooling strategy (default: %s)"%DEFAULT_POOLING)

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])

if __name__ == "__main__":
    sys.exit(main())

