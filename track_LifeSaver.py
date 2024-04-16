import os
import cv2
import argparse
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

HOME = os.getcwd()

from typing import List
import numpy as np

import supervision
from supervision.geometry.core import Point
from supervision import get_video_frames_generator
from supervision import VideoInfo
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.line_counter import LineZoneAnnotator

# from counter_utils import LineZoneLogger, point_inside_polygon, compute_max_min_line, compute_polygon_shapes
from tqdm.notebook import tqdm

from datetime import datetime, timedelta
import csv
ultralytics.checks()
print("supervision.__version__:", supervision.__version__)

import pandas as pd
import matplotlib.pyplot as plt

refPt = []

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='People Counter App')
    parser.add_argument('--p', default='', type=str,
                        help='Path for video folder'
                        )

    parser.add_argument('--out_vid', required=False, help=' (optional) Output video file')

    parser.add_argument('--log', default='', type=str,
                        help='log file'
                        )
    parser.add_argument('--vid_log_csv_path', default='', type=str,
                        help='path to read the logs obtained'
                        )
    
    parser.add_argument('--model_path', default='yolov8n.pt', type=str,
                        required=False, help=' (optional) Provide path to model. Default <yolov8n.pt>')

    args = parser.parse_args()

    return args
                
######### Session (Start) #########
def main():

    MODEL = "yolov8n.pt"

    parent_path = 'DesignA'
    output_path = 'DesignA_Output/'

    model = YOLO(MODEL)
    face_model = YOLO('yolov8n-face.pt')
    model.fuse()
    # print(model.names)
    # dict maping class_id to class_name
    # print(model.model.names)
    CLASS_NAMES_DICT = model.model.names

    # class_ids of interest - People
    CLASS_ID = [0]
    

    #path_list = ['16_2_2024 11_00_03 AM (UTC+08_00).mp4']
    df = pd.DataFrame()
    df['People_ID'] = None
    df['Timestamp'] = None
    df['Camera Number'] = None
    df['Body File Path'] = None
    df['Face File Path'] = None

    # df['People_ID'] = people_id
    # df['Timestamp'] = timestamp
    print(df)


    for day_night in os.listdir(parent_path):
        path1 = parent_path+'/'+day_night
        for scenario in os.listdir(path1):
            scenario_path = path1+'/'+scenario
            for camera_type in os.listdir(scenario_path+'/Video/'):
                temp = camera_type.split('_')
                print(temp)
                temp1 = temp[2].split('-')

                # start_time = temp1[0]
                # end_time = temp1[1]
                # print(start_time,end_time)


                

                # # Parse the time string
                # start_time_obj = datetime.strptime(start_time, '%I.%M %p').time()

                # # Combine with today's date to create a datetime object
                # start_time = datetime.combine(datetime.now().date(), start_time_obj)
                # end_time_obj = datetime.strptime(start_time, '%I.%M %p').time()

                # # Combine with today's date to create a datetime object
                # end_time = datetime.combine(datetime.now().date(), end_time_obj)

                camera_path = scenario_path+'/Video/'+camera_type
                for file in os.listdir(camera_path):
                    if 'avi' in file:
                        print()
                        camera_id = []
                        people_ids = []
                        timestamp = []
                        id_store = []
                        body_filepaths = []
                        face_filepaths = []
                        num_frame = 0
                        camera_id_num = temp[0]+'_'+temp[1]
                        vid_path = camera_path+'/'+file #'NVR 56_IPCamera 04_20240222163459_20240222163859.avi'

        
                        generator = get_video_frames_generator(vid_path) # '16_2_2024 11_00_03 AM (UTC+08_00).mkv')#
                        name_split = file.split('_')
                        date_string = name_split[2]

                        year  = date_string[:4]
                        month = date_string[4:6]
                        day   = date_string[6:8]
                        hour = date_string[8:10]
                        minute = date_string[10:12]
                        second = date_string[12:]
                        print((int(year), int(month), int(day), int(hour), int(minute), int(second)))
                        current_datetime = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

                        max_fps = 20 # 20 frames per 2 seconds

                        # print(video_info)
                        # print(max_fps)
        


                        camera_id = []
                        people_ids = []
                        timestamp = []
                        id_store = []
                        num_frame = 0
                        model_inference= YOLO('yolov8n.engine')
                        last_time = None

                        for frame in generator:
                            # print(current_datetime)
                            num_frame+=1
                            if num_frame >20:
                                num_frame=1
                                current_datetime += timedelta(seconds=2)

                            # model prediction on single frame and conversion to supervision Detections
                            #results = model.predict(frame, device=0, verbose=True)

                            # results = model.track(frame, persist=True, conf=0.1, tracker='config/trackers/bytetracker.yaml')
                            try:
                                results = model_inference.track(frame, persist=True, conf=0.1, tracker='config/trackers/bytetracker.yaml')
                                #filter results that cls is CLASS_ID (human)
                                #print(results[0].boxes)
                                results[0] = results[0][np.isin(results[0].boxes.cls.cpu().tolist(), CLASS_ID)]
                                
                                detections = supervision.Detections(
                                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                                    confidence=results[0].boxes.conf.cpu().numpy(),
                                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                                )

                                # Extract prediction results
                                boxes = results[0].boxes.xyxy.cpu().tolist()
                                print(boxes)
                                clss = results[0].boxes.cls.cpu().tolist()
                                track_ids = results[0].boxes.id.int().cpu().tolist()
                                #confs = results[0].boxes.conf.float().cpu().tolist()

                                detections.tracker_id = results[0].boxes.id.int().cpu().numpy()

                            except:
                                print('---------------------------------------------------------------')
                                track_ids = []

                            #annotator = Annotator(frame, line_width=2)
                            if len(track_ids) == 0:
                                continue
                            else:
                                
                                if num_frame%5 ==0: ##(current_datetime.second%2==0) and last_time != current_datetime: 
                                    last_time = current_datetime

                                    for box, cls, track_id in zip(boxes, clss, track_ids):
                                        x1 =  int(box[0])
                                        y1 = int(box[1])
                                        x2 = int(box[2])
                                        y2 = int(box[3])

                                        bodyframe = frame[y1:y2, x1:x2]

                                        results = face_model.predict(bodyframe, device=0, verbose=True)
                                        index = 1
                                        dest_path = 'results_face/'+ scenario_path+'/Video/'+camera_type
                                        time_str = str(current_datetime).replace(':',"_")
                                        # Visualize the detections
                                        if not os.path.exists(dest_path):
                                            os.makedirs(dest_path)
                                        highest_conf =0
                                        face_img = 'nil'
                                        for p in range(len(results[0].boxes.xyxy)):
                                            detection = results[0].boxes.xyxy[p]
                                            conf = results[0].boxes.conf[p]
                                            print(detection, conf)
                                            x_min, y_min, x_max, y_max = detection[0], detection[1], detection[2], detection[3]
                                            x_min = int(detection[0])
                                            y_min = int(detection[1])
                                            x_max = int(detection[2])
                                            y_max = int(detection[3])
                                            # if float(conf) >0.75:
                                            print(conf)
                                            if highest_conf<conf:
                                                face_img = bodyframe[y_min:y_max, x_min:x_max] 
                                        if face_img == 'nil':
                                            body_filepaths.append('nil')
                                        else:
                                            cv2.imwrite(dest_path+'/'+str(track_id)+'_'+camera_id_num+'_'+str(time_str)+'_'+str(num_frame)+'.png', face_img)
                                            body_filepaths.append(dest_path+'/'+str(track_id)+'_'+camera_id_num+'_'+str(time_str)+'_'+str(num_frame)+'.png')

                                        person_image = frame[y1:y2, x1:x2]
                                        folder_name = 'results_body/'+ scenario_path+'/Video/'+camera_type
                                        if not os.path.exists(folder_name):
                                            os.makedirs(folder_name)
                                        cv2.imwrite(folder_name+'/'+str(track_id)+'_'+camera_id_num+'_'+str(time_str)+'_'+str(num_frame)+'.png', person_image)
                                        people_ids.append(track_id)
                                        camera_id.append(camera_id_num)
                                        timestamp.append(str(time_str))
                                        body_filepaths.append(folder_name+'/'+str(track_id)+'_'+camera_id_num+'_'+str(time_str)+'_'+str(num_frame)+'.png')
                                        
                                        

                                    
            # cv2.imshow("", frame) 
            # cv2.waitKey(1)
        df['Camera Number'] = camera_id
        df['People_ID'] = people_ids
        df['Timestamp'] = timestamp
        df['Body File Path'] = body_filepaths
        df['Face File Path'] = face_filepaths
        print(df)
        df.to_csv('results/FONDA'+day_night+'.csv')

    #             line_counter.trigger(detections=detections, logfile=logfile)
                


    #             #draw the line and count the stats
    #             line_annotator.annotate(frame=frame, line_counter=line_counter)


    #             #Time detection and Recording to CSV
                
    #             num_frame, logged_status , current_datetime,last_interval,df = record_csv(num_frame,current_datetime,logged_status,line_annotator,file_path,line_counter,df)
                
    #         except: 
    #             num_frame, logged_status , current_datetime,last_interval,df  = record_csv(num_frame,current_datetime,logged_status,line_annotator,file_path,line_counter,df)
    #         #sink.write_frame(frame)
    #         if args.out_vid:
    #             vidWriter.write(frame)



######### Session (End) #########
main()  

#python track_people_w_gui.py --v people.mp4 --log counter.csv