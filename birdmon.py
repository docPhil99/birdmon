import cv2
import sys
import imutils
import csv
import datetime
import logging
import json
import argparse
from pathlib import Path
import math
from imutils.video import FileVideoStream

logger = logging.getLogger(__name__)
# ffmpeg -i 2017_0107_005726_017.MP4  -filter:v "fps=15, scale=640:-1" smaller.mp4


def _save_json(name,config):
    with open(name,mode='w') as jf:
        json.dump(config, jf)

def bb_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s: %(message)s')

my_parser = argparse.ArgumentParser(description='Birdmon')
my_parser.add_argument('video_file_name', type=Path, help='Video file name')
my_parser.add_argument('-output_dir', type=Path, default=Path('output'), help="output directory, defaults to 'output/'")


my_parser.add_argument('-mask_box_coords',type=int, nargs=4, default=None,
                help='Coordinates of mask box, if set the GUI bounding box will not be used. Format is x1,y1, x2,y2')
my_parser.add_argument('-min_area_percent',type=float ,default=0.7, help='Min percentage area of the image taken by the bird, default 0.7')
my_parser.add_argument('-csv_file_name',type=Path, default=None, help='output CSV file name, if not set it will be based on the video filename')
my_parser.add_argument('-min_frame_alarm_count',type=float, help='How many frames to track the bird before triggering an event, default is 4', default=4)
my_parser.add_argument('-log_file_name',type=Path, default=Path('log.txt'), help='log file name, defaults to log.txt')
my_parser.add_argument('-resize',action='store_true', help='Resizes the images to 640x? and 15 fps (ish)')
my_parser.add_argument('-width',default=640, type=int, help='If resize flag is set, use this value for width of the image, default is 640')


# Execute the parse_args() method
config = vars(my_parser.parse_args())

fileHandler = logging.FileHandler(config['log_file_name'],'w+')
logger.addHandler(fileHandler)

if config['csv_file_name'] is None:
    config['csv_file_name'] = config['output_dir'] / Path(config['video_file_name'].stem).with_suffix('.csv')

logger.info(f"Output CSV file: {config['csv_file_name']}")

# make the output directory
config['output_dir'].mkdir(parents= True, exist_ok= True)

if config['mask_box_coords'] is not None:
    config['mask_box_coords'] = ((config['mask_box_coords'][0], config['mask_box_coords'][1]),
                                            (config['mask_box_coords'][2],config['mask_box_coords'][3]))

logger.debug(f'Config: {config}')
fvs = FileVideoStream(str(config['video_file_name'])).start()
#cap = cv2.VideoCapture(str(config['video_file_name']))
if not fvs.stream.isOpened():
    logger.error(f"Can not open file {config['video_file_name']}")
    sys.exit(-1)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_rate = fvs.stream.get(cv2.CAP_PROP_FPS)
logger.info(f'Frame rate {frame_rate}')
skip_rate = math.ceil(frame_rate/15)
frame_alive_counter = 0
bird_id =0

try:
    with open(config['csv_file_name'], mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        frame_counter = 0
        csv_writer.writerow(['Bird ID', 'Frame Number', 'Frame Time'])
        while fvs.more():
            #ret, frame = cap.read()
            frame = fvs.read()
            if frame is None:
                logger.warning('Bad frame error - aborting. This might be the end of the file')
                sys.exit(1)
            if config['resize']:
                frame = imutils.resize(frame, width=config['width'])

            if frame_counter == 0 and config['mask_box_coords'] is None:
                x,y,w,h = cv2.selectROI("ROI", frame)
                config['mask_box_coords'] = ((x,y), (x+w, y+h))
                logger.info(f"gui selected ROI: {config['mask_box_coords']}")

            if frame_counter == 0:

                image_area = frame.shape[0]*frame.shape[1]
                min_area = config['min_area_percent']*image_area/100
                logger.info(f'Image size {frame.shape} Image area {image_area}, min bird size {min_area}')

            frame_counter += 1
            if frame_counter % skip_rate != 0:
                #print('skip')
                continue
            #print('process')

            fgmask = fgbg.apply(frame)

            thresh = cv2.dilate(fgmask, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.rectangle(frame, tuple(config['mask_box_coords'][0]), tuple(config['mask_box_coords'][1]), (255, 255, 0), 2)
            # loop over the contours
            active_frame_counter = 0
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                intersect = bb_intersection((x,y,x+w,y+h),tuple(element for tupl in config['mask_box_coords'] for element in tupl))
                if intersect > 0:
                    #logger.debug(intersect)
                    col = (255, 0, 0)
                    active_frame_counter += 1
                else:

                    col = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)

                cv2.putText(frame, f'Area: {cv2.contourArea(c)}', (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.35, col,1)
            if active_frame_counter > 0:
                frame_alive_counter += 1
            else:
                frame_alive_counter = 0
            output = [bird_id, frame_counter, str(datetime.timedelta(seconds=frame_counter / frame_rate))]
            if frame_alive_counter == 1:
                bird_id += 1
                csv_writer.writerow(output)

            if frame_alive_counter == config['min_frame_alarm_count']:
                cv2.putText(frame, f'Alert {active_frame_counter}', (20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7, col,1)

                logger.debug(f'alert bird ID {output}')

                # save image
                fname = config['video_file_name'].stem
                img_path = config['output_dir'] / Path(f'{fname}_{bird_id}.jpg')
                logger.debug(f'Saving {img_path}')
                cv2.imwrite(str(img_path), frame)

            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except:
    logger.exception('Failed to run')
#cap.release()
cv2.destroyAllWindows()
