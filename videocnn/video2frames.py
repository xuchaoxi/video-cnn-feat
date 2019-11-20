import os, sys
import time

import cv2
import logging

from constant import ROOT_PATH, PROGRESS

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def process(options, collection):
    rootpath = options.rootpath
    overwrite = options.overwrite

    id_path_file = os.path.join(rootpath, collection, 'id.videopath.txt')
    data = map(str.strip, open(id_path_file).readlines())
    videoset = [x.split()[0] for x in data]
    filenames = [x.split()[1] for x in data]

    output_dir = os.path.join(rootpath, collection, 'ImageData')
    num_of_videos = len(videoset)

    total_frame_count = 0
    records = {}

    for i in range(num_of_videos):
        video_file = filenames[i]
        video_id = videoset[i]
        if i % PROGRESS == 0:
            logger.info('extracting frames from video %d / %d: %s' % (i, num_of_videos, video_id))
        frame_output_dir = os.path.join(output_dir, videoset[i])
        if os.path.exists(frame_output_dir) and not overwrite:
            logger.debug('Skipping video: %s'% videoset[i])
            continue
        if not os.path.exists(frame_output_dir):
            os.makedirs(frame_output_dir)

        cap = cv2.VideoCapture(video_file)
        if cv2.__version__.startswith('3'):
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        else:
            length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        records[video_id] = (fps, length, width, height)
 
        flag = True
        fcount = 0
        flag, frame = cap.read()
        while(flag):
            # Write the frame every 0.5 second
            if fcount % fps == 0 or fcount % fps == (fps/2):
                cv2.imwrite(os.path.join(frame_output_dir, '%s_%d.jpg'%(video_id, fcount)), frame)
                total_frame_count += 1
            fcount += 1
            flag, frame = cap.read()
       
        if fcount > 0: 
            records[video_id] = (fps, length, width, height)
        else:
            logger.error("failed to process %s", video_id)

    # record fps, frame_count, weight, height per video
    video_meta_file = os.path.join(rootpath, collection, 'id.videometa.txt')
    fw = open(video_meta_file, 'w')
    for video_id in videoset:
        if video_id in records:
            fps, length, width, height = records[video_id]
            fw.write('%s %d %d %d %d\n' % (video_id, fps, length, width, height))
    fw.close()

    logger.info("%d videos -> %d frames extracted", num_of_videos, total_frame_count)

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


if __name__ == '__main__':
    sys.exit(main())

