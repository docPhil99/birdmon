import cv2
import sys
import imutils
import csv
import datetime
import logging
import json
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)
# ffmpeg -i 2017_0107_005726_017.MP4  -filter:v "fps=15, scale=640:-1" smaller.mp4


def _save_json(name,config):
    with open(name,mode='w') as jf:
        json.dump(config,jf)




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
my_parser.add_argument('config', type=str,  help='config json file name')

# Execute the parse_args() method
args = my_parser.parse_args()

config_path = args.config
logger.debug(f'Loading config file {config_path}')

try:
    with open(config_path) as jf:
        config = json.load(jf)
except:
    logger.exception(f'Cannot load {config_path}')
    sys.exit(-1)

logger.debug(f'Config: {config}')
#config= {'video_file_name':'/home/phil/Datasets/birds/smaller_test.mp4','csv_filename': 'test.csv', 'min_area_percent': 0.7, 'keep_alive': 3 ,'min_frame_alarm_count':4,
#    'mask_box_coords':((200, 100), (360, 330)) }


#cap = cv2.Vide'oCapture('/home/phil/Datasets/birds/trimmed_bird1.mp4')
cap = cv2.VideoCapture(config['video_file_name'])
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_rate = cap.get(cv2.CAP_PROP_FPS)
logger.info(f'Frame rate {frame_rate}')



frame_alive_counter = 0
bird_id =0

try:
    with open(config['csv_filename'], mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        frame_counter = 0
        csv_writer.writerow(['Bird ID', 'Frame Number', 'Frame Time'])
        while True:
            ret, frame = cap.read()
            if frame is None:
                logger.warning('Bad frame error - aborting. This might be the end of the file')
                sys.exit(1)
            if frame_counter == 0:
                roi = cv2.selectROI("ROI", frame)
                logger.info(f'gui selectedf ROI: {roi}')
            frame_counter += 1
            fgmask = fgbg.apply(frame)
            if frame_counter == 1:

                image_area = frame.shape[0]*frame.shape[1]
                min_area = config['min_area_percent']*image_area/100
                logger.info(f'Image size {frame.shape} Image area {image_area}, min bird size {min_area}')
            #print(image_area,min_area)

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
                fname = Path(config_path).stem
                img_path = Path(config['save_img_dir']) / Path(f'{fname}_{bird_id}.jpg')
                logger.debug(f'Saving {img_path}')
                cv2.imwrite(str(img_path), frame)

            cv2.imshow('frame',frame)
            #print(frame_alive_counter)
            #cv2.imshow('mask',fgmask)
            #cv2.imshow('mask', thresh)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except:
    logger.exception('Failed to run')
cap.release()
cv2.destroyAllWindows()
