#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : zhichao
# video face recognition : include video segmentation based on image similarity, predicted picture with Face recognition bounding box, by video face recognition scheme, I obtain *.csv file content which has been sorted.

from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
# build-in function
import os
import math
import codecs
import re
import time
import cv2
import json
import numpy as np
import face
global duration_time
global fps

face_recognition = face.Recognition()
ad_collector = []

# threshold --- compute similarity
def get_similarity(img1, img2):

    # initialization
    degree = 0

    # click calcHist have a question # [1]
    hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 255])
    hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 255])
    length = len(hist_img1)

    for i in range(length):
        if hist_img1[i] != hist_img2[i]:
            degree = degree + (1 - abs(hist_img1[i] - hist_img2[i]) / max(hist_img1[i], hist_img2[i]))
        else:
            degree = degree + 1

    degree = degree / length

    return degree

# Compute time during the process
def times_formatting(seconds):

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 0 and hours < 10:
        hours_formatting = '0' + str(int(hours))
    else:
        hours_formatting = str(int(hours))
    if minutes >= 0 and minutes < 10:
        minutes_formatting = '0' + str(int(minutes))
    else:
        minutes_formatting = str(int(minutes))
    if seconds >= 0 and seconds < 10:
        seconds_formatting = '0' + str(int(seconds))
    else:
        seconds_formatting = str(int(seconds))

    return hours_formatting, minutes_formatting, seconds_formatting

def video_parser(filepath):

    # import face
    global duration_time
    global fps
    global ad_collector
    duration_time = []
    generate_file_path = "./describe/advertise/"
    if not os.path.exists(generate_file_path):
        os.makedirs(generate_file_path)
    # shot stage
    start = time.time()
    face_cascade = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
    result_json = []
    ad_collector = []
    print("视频分析实验平台分镜处理")
    # param set : the num of frame in every shot
    # video Capture
    videoCapture = cv2.VideoCapture(filepath)
    file_name_ext = re.split(r'/', filepath)[-1]
    file_name = file_name_ext.split('.')[0]
    # obtain fps
    fps = math.floor(videoCapture.get(cv2.CAP_PROP_FPS))
    exceed_3s = 3 * fps

    nFrames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    success1, frame1 = videoCapture.read()

    start_frame_duration = 1
    end_frame_duration = 1

    frame = frame1
    exchange_count = 0
    interval_count = 0
    tip = 8
    shot_all_img = []
    threshlod1 = 0.85
    threshlod2 = 0.75

    for i in range(nFrames - 1):
        flag = False

        cur_image_id = i + 1
        exchange_count += 1
        interval_count += 1

        success2, frame2 = videoCapture.read()

        end_frame_duration += 1

        # storage all mat_img in sub-shot
        shot_all_img.append(frame2)

        neighbor_similar = get_similarity(frame1, frame2)
        if exchange_count > tip:
            exchange_count = 0
            frame = frame2
        distantRelative_similar = get_similarity(frame, frame2)
        if (neighbor_similar < threshlod1) | (distantRelative_similar < threshlod2):

            # The method which named from start frame to end frame in a shot
            if (interval_count > exceed_3s):
                for num in range(interval_count):
                    if num == 0:
                        continue
                    if num % 1 == 0:
                        img_symbol_shot = shot_all_img[num]
                        gray = cv2.cvtColor(img_symbol_shot, cv2.COLOR_BGR2GRAY)
                        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

                        if len(faces_detected) > 0:
                            faces = face_recognition.identify(img_symbol_shot)
                            with codecs.open("result.json", "w", encoding="utf-8") as ad_spots:
                                for face in faces:
                                    face_bb = face.bounding_box.astype(int)

                                    if face.confidence > 0.50:
                                        result_json.append(
                                            {"label": str(face.name), "confidence": str(face.confidence),
                                             "topleft": {"x": str(face_bb[0]), "y": str(face_bb[1])}, "bottomright": {"x": str(face_bb[2]), "y": str(face_bb[3])}}
                                        )
                                        for i in range(30):
                                            img_symbol_shot[face_bb[1] + i][face_bb[0]] = 255
                                            img_symbol_shot[face_bb[1] + i][face_bb[0] - 1] = 255
                                            img_symbol_shot[face_bb[1] + i][face_bb[0] + 1] = 255
                                            img_symbol_shot[face_bb[1]][face_bb[0] + i] = 255
                                            img_symbol_shot[face_bb[1] - 1][face_bb[0] + i] = 255
                                            img_symbol_shot[face_bb[1] + 1][face_bb[0] + i] = 255

                                            img_symbol_shot[face_bb[3] - i][face_bb[2]] = 255
                                            img_symbol_shot[face_bb[3] - i][face_bb[2] - 1] = 255
                                            img_symbol_shot[face_bb[3] - i][face_bb[2] + 1] = 255
                                            img_symbol_shot[face_bb[3]][face_bb[2] - i] = 255
                                            img_symbol_shot[face_bb[3] - 1][face_bb[2] - i] = 255
                                            img_symbol_shot[face_bb[3] + 1][face_bb[2] - i] = 255

                                            img_symbol_shot[face_bb[1] + i][face_bb[2]] = 255
                                            img_symbol_shot[face_bb[1] + i][face_bb[2] - 1] = 255
                                            img_symbol_shot[face_bb[1] + i][face_bb[2] + 1] = 255
                                            img_symbol_shot[face_bb[1]][face_bb[2] - i] = 255
                                            img_symbol_shot[face_bb[1] - 1][face_bb[2] - i] = 255
                                            img_symbol_shot[face_bb[1] + 1][face_bb[2] - i] = 255

                                            img_symbol_shot[face_bb[3] - i][face_bb[0]] = 255
                                            img_symbol_shot[face_bb[3] - i][face_bb[0] - 1] = 255
                                            img_symbol_shot[face_bb[3] - i][face_bb[0] + 1] = 255
                                            img_symbol_shot[face_bb[3]][face_bb[0] + i] = 255
                                            img_symbol_shot[face_bb[3] - 1][face_bb[0] + i] = 255
                                            img_symbol_shot[face_bb[3] + 1][face_bb[0] + i] = 255


                                        frame_rgb = cv2.cvtColor(img_symbol_shot, cv2.COLOR_BGR2RGB)
                                        frame_pil = Image.fromarray(frame_rgb)
                                        draw = ImageDraw.Draw(frame_pil)
                                        font = ImageFont.truetype("SimHei.ttf", 35, encoding="utf-8")
                                        draw.text((face_bb[0], face_bb[3] + 5), face.name, (0, 0, 255), font=font)
                                        frame_text = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                                        img_symbol_shot = np.array(frame_text)

                                        flag = True

                                text_in = json.dumps(result_json)
                                ad_spots.write(text_in)
                                print('-----------------正在进行视频人脸定位，请稍后---------------')

                        if flag == True:
                            image_selected_pos = cur_image_id - (interval_count - num - 1)

                            if not os.path.isdir("./img/" + str(file_name)):
                                os.makedirs("./img/" + str(file_name))
                            cv2.imencode('.jpg', img_symbol_shot)[1].tofile('./img/' + str(file_name) + '/' + str(file_name) + '_' + str(image_selected_pos) + 'face.jpg')

                            ad_generator(file_name, start_frame_duration, end_frame_duration, image_selected_pos)
                            break

                result_json = []

            exchange_count = 1
            frame = frame2
            interval_count = 0
            shot_all_img = []

            start_frame_duration = end_frame_duration

        frame1 = frame2

    end = time.time()
    seconds = end - start
    hours_formatting, minutes_formatting, seconds_formatting = times_formatting(seconds)

    print("视频分析完成，消耗时长: %s : %s : %s" % (hours_formatting, minutes_formatting, seconds_formatting))
    sort_by_weight(file_name)
    generate_stdFile(file_name, generate_file_path + file_name)

# generate final file for tester
def generate_stdFile(file_name, path):

    final_file = path + "_" + "hmzc" + ".csv"
    with codecs.open(path, 'r', encoding='utf-8') as normal_file:
        foot = 4

        multiLines = normal_file.readlines()
        with codecs.open(final_file, 'w+', encoding='utf-8') as std_file:

            std_file.write(
                "片名" + "," + "入点时间" + "," + "出点时间" + "," + "片段时长" + "," + "广告位类别" + "," + "置信度"
            )

            std_file.write("\n")
            for line in multiLines:
                start_index = 2

                words = line.split(',')
                start_time = words[0].split('--')[0]
                end_time = words[0].split('--')[1]

                s_time = datetime.strptime(start_time, '%H:%M:%S')
                e_time = datetime.strptime(end_time, '%H:%M:%S')
                elapse_time = e_time - s_time
                while words[start_index] != 'None':
                    std_file.write(
                        file_name + '_' + words[1].split("_")[0] + "face" + ',' + start_time + ',' + end_time + ',' + str(elapse_time) + ',' + words[start_index] + ',' + words[start_index+1]
                    )
                    start_index += foot
                    std_file.write("\n")

        print("{}文件生成，请查看".format(os.path.split(final_file)[1]))
    return
# translate shot into word, with date, time, frame_id
def ad_generator(file_name, start_frame_duration, end_frame_duration, image_selected_pos):
    # the pos of advertisement detect
    global fps

    with open('result.json', 'r') as ad_finder:
        # temp list for a line for a frame det
        records = []

        ad_record = ad_finder.readline()

        empty_sit = 10

        time_plan_start = start_frame_duration / fps
        time_plan_end = end_frame_duration / fps
        start_h, start_m, start_s = times_formatting(time_plan_start)
        end_h, end_m, end_s = times_formatting(time_plan_end)

        if len(ad_record) <= 3:
            pass

        else:

            ad_record = list(eval(ad_record))

            records.append(
                start_h + ":" + start_m + ":" + start_s + "--" +
                end_h + ":" + end_m + ":" + end_s
            )
            records.append(
                str(image_selected_pos) + '_' + str(start_frame_duration) + '-' + str(end_frame_duration)
            )
            # records.append(str(frame_id_img))
            # print("ad_record = " + str(ad_record))
            for ad_parser in ad_record:

                empty_sit -= 1

                # list struct
                records.append(str(ad_parser['label']))
                records.append(str(ad_parser['confidence']))
                records.append(
                    'x:' + str(ad_parser['topleft']['x']) + ' ' + 'y:' + str(
                    ad_parser['topleft']['y'])
                )
                records.append(
                    'x:' + str(ad_parser['bottomright']['x']) + ' ' + 'y:' + str(
                    ad_parser['bottomright']['y'])
                )

            for i in range(empty_sit):

                records.append('None')
                records.append('None')
                records.append('None')
                records.append('None')
            ad_collector.append(records)

    return

def sort_by_weight(filename):
    foot_add = 4
    max_array = []
    ad_appeared_disorder = {}
    all_info_sort = []
    all_info_final = []

    with codecs.open("./describe/advertise/" + filename, "w+", encoding="utf-8") as sort_writer:
        for outer_num_list in range(len(ad_collector)):
            if ad_collector[outer_num_list][2] is not 'None':
                all_info_sort.append(ad_collector[outer_num_list])

        # for None_info in None_info_backup:
        for outer_num_list in range(len(all_info_sort)):

                conf_pos = 3
                end_solider = 40
                max_value = 0
                while conf_pos < end_solider:
                    if all_info_sort[outer_num_list][conf_pos] != 'None':
                        # print(all_info_sort[outer_num_list][conf_pos])
                        conf_value = float(all_info_sort[outer_num_list][conf_pos])
                        # print("conf_value" + str(conf_value))
                        if conf_value > max_value:
                            max_value = conf_value
                    else:
                        break
                    conf_pos += foot_add
                max_array.append(max_value)

        for i in range(len(max_array)):
            ad_appeared_disorder[i] = max_array[i]

        ad_appeared_order = sorted(ad_appeared_disorder.items(), key=lambda e : e[1], reverse=True)

        for key in range(len(ad_appeared_order)):
            idn = ad_appeared_order[key][0]
            all_info_final.append(all_info_sort[idn])

        for info in all_info_final:
            for num in range(len(info)):
                if num < len(info) -1:
                    sort_writer.write(info[num] + ",")
                else:
                    sort_writer.write(info[num])

            sort_writer.write("\n")

    return

if __name__ == "__main__":

    # add video path
    path = "./video/test.avi"
    video_parser(path)
