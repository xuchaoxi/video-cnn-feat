'''
concat several features (axis=1)
'''

import os
import sys
import logging
from constant import ROOT_PATH
from utils.generic_utils import Progbar
from utils.bigfile import BigFile

logger = logging.getLogger(__file__)
logging.basicConfig(
        format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
        datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def process(options, collection, featnames):
    rootpath = options.rootpath
    target_featname = featnames
    featnames = featnames.split('+')
    target_feat_dir = os.path.join(rootpath, collection, 'FeatureData', target_featname)

    if os.path.exists(target_feat_dir):
        if options.overwrite:
            logger.info('%s exists! overwrite.', target_feat_dir)
        else:
            logger.info('%s exists! quit.', target_feat_dir)
            sys.exit(0)
    else:
        os.makedirs(target_feat_dir)

    target_feat_file = os.path.join(target_feat_dir, 'id.feature.txt')
    target_id_file = os.path.join(target_feat_dir, 'id.txt')

    target_feats_vec = {}
    img_ids = []

    with open(target_feat_file, 'w') as fw_feat, open(target_id_file, 'w') as fw_id:
        for feat in featnames:
            logger.info('>>> Process %s...', feat)
            feat_dir = os.path.join(rootpath, collection, 'FeatureData', feat)
            featfile = BigFile(feat_dir)
            renamed, vectors = featfile.readall()

            if not target_feats_vec:
                target_feats_vec = dict(zip(renamed, vectors))
                img_ids = renamed
                assert(len(img_ids) == len(set(img_ids))), '%s contain duplicated img ids'%feat
            else:
                assert(len(renamed) == len(img_ids) and set(img_ids) == set(renamed)), '%s not match target feature'%feat
                for name, vec in zip(renamed, vectors):
                    target_feats_vec[name].extend(vec)

        logger.info('>>> Save to %s', target_feat_dir)
        fw_id.write(' '.join(img_ids))
        progbar = Progbar(len(target_feats_vec))
        for name, feat in target_feats_vec.iteritems():
            fw_feat.write('%s %s\n' % (name, ' '.join(['%g'%x for x in feat])))
            progbar.add(1)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection featnames""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])

if __name__ == '__main__':
    sys.exit(main())

