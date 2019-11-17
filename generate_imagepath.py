import sys, os
import logging
from constant import ROOT_PATH

FILTER_SET = set(str.split(".jpg"))

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

    
def process(options, collection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    
    resultfile = os.path.join(rootpath, collection, "id.imagepath.txt")
    
    if os.path.exists(resultfile):
        if not overwrite:
            logger.info("%s exists. quit", resultfile)
            return 0
        else:
            logger.info("%s exists. overwrite", resultfile)

    imageFolders = [os.path.join(rootpath, collection, 'ImageData')]
    filenames = []
    imageset = set()
    
    for imageFolder in imageFolders:
        for r,d,f in os.walk(imageFolder):
            for filename in f:
                if collection == 'tgif':
                    if int(filename.split('.')[0].split('_')[2]) % 4 != 0:   # generate frames every 4 frames for tgif
                        continue
                name,ext = os.path.splitext(filename)
                if ext not in FILTER_SET:
                    continue

                if collection.startswith('activitynet'):
                    name = os.path.basename(r)+'_'+name
                    
                if name in imageset:
                    print ("id %s exists, ignore %s" % (name, os.path.join(r,filename)))
                    continue
                    
                imageset.add(name)
                filenames.append("%s %s" % (name, os.path.join(r, filename)))

    with open(resultfile, "w") as fout:
        fout.write("\n".join(filenames) + "\n")
    
    idfile = os.path.join(rootpath, collection, "ImageSets", '%s.txt' % collection)
    try:            
        os.makedirs(os.path.split(idfile)[0])
    except:
        pass

    with open(idfile, "w") as fout:
        fout.write("\n".join(sorted(list(imageset))) + "\n")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])

if __name__ == "__main__":
    sys.exit(main())

