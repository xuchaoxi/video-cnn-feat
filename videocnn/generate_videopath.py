import sys, os
import logging
from constant import ROOT_PATH

FILTER_SET = set(str.split(".mp4 .avi .webm .gif"))

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

    
def process(options, collection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    
    resultfile = os.path.join(rootpath, collection, "id.videopath.txt")
    
    if os.path.exists(resultfile):
        if overwrite:
            logger.info("%s exists. overwrite", resultfile)
        else:
            logger.info("%s exists. quit", resultfile)
            return 0

    videoFolders = [os.path.join(rootpath, collection, 'VideoData')]  # VideoData
    filenames = []
    videoset = set()
    
    for videoFolder in videoFolders:
        for r,d,f in os.walk(videoFolder):
            for filename in f:
                name,ext = os.path.splitext(filename)
                if ext not in FILTER_SET:
                    continue
                    
                if name in videoset:
                    print ("id %s exists, ignore %s" % (name, os.path.join(r,filename)))
                    continue
                    
                videoset.add(name)
                filenames.append("%s %s" % (name, os.path.join(r, filename)))

    with open(resultfile, "w") as fout:
        fout.write("\n".join(filenames) + "\n")
    
    idfile = os.path.join(rootpath, collection, "VideoSets", '%s.txt' % collection)
    try:            
        os.makedirs(os.path.split(idfile)[0])
    except:
        pass           
    with open(idfile, 'w') as fout:
        fout.write("\n".join(sorted(list(videoset))) + "\n")


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

