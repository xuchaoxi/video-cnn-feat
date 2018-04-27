'''
join several sub collections' features into one large collection feature
'''

import os
import sys
import logging
from generic_utils import Progbar
from constant import ROOT_PATH
from bigfile import BigFile

logger = logging.getLogger(__file__)
logging.basicConfig(
        format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
        datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def process(options, collection, featname, sub_collections):
    rootpath = options.rootpath
    target_feat_dir = os.path.join(rootpath, collection, 'FeatureData', featname)
    target_img_file = os.path.join(rootpath, collection, 'ImageSets', collection+'.txt')

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
    sub_collections = sub_collections.split('@')
    img_ids = [] 

    with open(target_feat_file, 'w') as fw_feat, open(target_id_file, 'w') as fw_id:
        for collect in sub_collections:
            feat_dir = os.path.join(rootpath, collect, 'FeatureData', featname)
            featfile = BigFile(feat_dir)
            renamed, vectors = featfile.readall()

            print(">>> Process %s" % collect)
            progbar = Progbar(len(renamed))
            for name, feat in zip(renamed, vectors):
                fw_feat.write('%s %s\n' % (name, ' '.join(['%g'%x for x in feat])))
                progbar.add(1)

            img_ids.extend(renamed)

        fw_id.write(' '.join(img_ids))

    if os.path.exists(target_img_file):
        current_ids = map(str.strip, open(target_img_file).readlines())
        if len(current_ids) != len(img_ids):
            logger.info('%s exists! but not match.', target_img_file)
            if options.overwrite:
                logger.info('overwrite %s.', target_img_file)
                with open(target_img_file, 'w') as fw_img:
                    fw_img.write('\n'.join(img_ids) + '\n')
            else:
                logger.info('quit %s.', target_img_file)
                return 0
    else:
        if not os.path.exists(os.path.dirname(target_img_file)):
            os.makedirs(os.path.dirname(target_img_file))
        with open(target_img_file, 'w') as fw_img:
            fw_img.write('\n'.join(img_ids) + '\n')
    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection featname [sub_collections]""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    return process(options, args[0], args[1], args[2])

if __name__ == '__main__':
    sys.exit(main())

