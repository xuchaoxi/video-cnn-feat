
import sys
import os
import logging
from constant import *
from bigfile import BigFile

logger = logging.getLogger(__file__)
logging.basicConfig(
        format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
        datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def process(options, collection, featname, sub_collections):
    rootpath = options.rootpath
    feat_dir = os.path.join(rootpath, collection, 'FeatureData', featname)

    sub_collections = sub_collections.split('@')
    featfile = BigFile(feat_dir)

    for collect in sub_collections:
        target_feat_dir = os.path.join(rootpath, collect, 'FeatureData', featname)
        target_img_file = os.path.join(rootpath, collect, 'ImageSets', collect+'.txt')
        target_img_list = [line.strip() for line in open(target_img_file)]
        renamed, vectors = featfile.read(target_img_list)
        assert len(target_img_list) == len(renamed)

        target_feat_dir = os.path.join(rootpath, collect, 'FeatureData', featname)
        if  os.path.exists(target_feat_dir):
            if options.overwrite:
                logger.info('%s exists! overwrite.', target_feat_dir)
            else:
                logger.info('%s exists! do nothing.', target_feat_dir)
                sys.exit(0)
        else:
            os.makedirs(target_feat_dir)
        target_feat_file = os.path.join(target_feat_dir, 'id.feature.txt')
        target_id_file = os.path.join(target_feat_dir, 'id.txt')
        fw_feat = open(target_feat_file, 'w')
        fw_id = open(target_id_file, 'w')
        for name, feat in zip(renamed, vectors):
            fw_feat.write('%s %s\n' % (name, ' '.join(['%g'%x for x in feat])))
        fw_feat.close()
        fw_id.write(' '.join(target_img_list))
        fw_id.close()


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
