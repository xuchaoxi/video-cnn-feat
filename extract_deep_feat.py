'''
Weiyu Lan, Xirong Li, Jianfeng Dong, Fluency-Guided Cross-Lingual Image Captioning, ACM MM 2017
'''

import sys
import os
import json
import time
import logging

from constant import *
from generic_utils import Progbar
from mxnet_feat_os2 import get_feat_extractor, extract_feature

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_feat_name(model_prefix, layer, oversample):
    if model_prefix.find('resnext-101_rbps13k')>=0:
        feat = 'resnext-101_rbps13k'
    elif model_prefix.find('resnext-101_places2')>=0:
        feat = 'resnext-101_places2'
    elif model_prefix.find('11k')>=0:
        feat = 'resnet-152_imagenet11k'
    else:
        feat = 'resnet-152_imagenet1k'
    return 'py%s,%s,os' % (feat,layer) if oversample else 'py%s,%s' % (feat, layer) 


def process(options, collection):
    rootpath = options.rootpath
    oversample = options.oversample
    model_prefix = os.path.join(rootpath, options.model_prefix)
    sub_mean = model_prefix.find('mxmodels80')>=0
    logger.info('subtract mean? %d', sub_mean)
    layer = 'flatten0_output'
    batch_size = 1 # change the batch size will get slightly different feature vectors. So stick to batch size of 1.
    feat_name = get_feat_name(model_prefix, layer, oversample)
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feat_name)
    id_file = os.path.join(feat_dir, 'id.txt')
    feat_file = os.path.join(feat_dir, 'id.feature.txt')

    for x in [id_file, feat_file]:
        if os.path.exists(x):
            if not options.overwrite:
                logger.info('%s exists. skip', x)
                return 0
            else:
                logger.info('%s exists. overwrite', x)

    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
    data = map(str.strip, open(id_path_file).readlines())
    img_ids = [x.split()[0] for x in data]
    filenames = [x.split()[1] for x in data]

    fe_mod = get_feat_extractor(model_prefix=model_prefix, gpuid=options.gpu, oversample=oversample)

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    feat_file = os.path.join(feat_dir, 'id.feature.txt')
    fw = open(feat_file,'w')

    im2path = zip(img_ids, filenames)
    success = 0
    fail = 0

    start_time = time.time()
    logger.info('%d images, %d done, %d to do', len(img_ids), 0, len(img_ids))
    progbar = Progbar(len(img_ids))

    for i, (imgid,impath) in enumerate(im2path):
        try:
            imid_list, features = extract_feature(fe_mod, 1, [imgid], [impath], sub_mean=sub_mean, oversample=oversample)
            for name, feat in zip(imid_list, features):
                fw.write('%s %s\n' % (name, ' '.join(['%g'%x for x in feat])))
            success += 1
        except:
            logger.error('failed to process %s', impath)
            logger.info('%d success, %d fail', success, fail)
            fail += 1
        finally:
            progbar.add(1)

        #if i%1e3 == 0:
        #    logger.info('%d success, %d fail', success, fail)

    logger.info('%d success, %d fail', success, fail)
    elapsed_time = time.time() - start_time
    logger.info('total running time %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
 
    fw.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--model_prefix", default=DEFAULT_MODEL_PREFIX, type="string", help=DEFAULT_MODEL_PREFIX)
    parser.add_option("--gpu", default=0, type="int", help="gpu id (default: 0)")
    parser.add_option("--oversample", default=1, type="int", help="oversample (default: 1)")
  
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    print json.dumps(vars(options), indent = 2)
    #assert(options.job>=1 and options.numjobs >= options.job)
    
    return process(options, args[0])


if __name__ == '__main__':
    sys.exit(main())

