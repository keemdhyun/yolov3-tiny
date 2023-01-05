#!/usr/bin/env python
#-*-coding: utf-8 -*-

#import rospy
import cv2
import youtube_dl
import pafy
import numpy as np
import matplotlib.pyplot as plt
import time

# 깊은 복사를 위해서
import copy

# ==================================================================================
# youtube 영상 불러오기
# https://velog.io/@bangsy/Python-OpenCV-4      설명
# https://www.youtube.com/watch?v=YsPdvvixYfo&t=0s
# https://www.youtube.com/watch?v=ipyzW38sHg0

# 원하는 유튜브 영상이나 아니면 그냥 cap = cv2.VideoCapture(0)으로 코드를 변경해서 웹캠으로 사용가능
url = 'https://www.youtube.com/watch?v=wgTyQGFrhgs'
video = pafy.new(url)
best = video.getbest(preftype = 'mp4')

# cap = cv2.VideoCapture(best.url)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('project_video.mp4')

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# 수정해야할 부분
net = cv2.dnn.readNet("st2.weights", "yolov3-tiny.cfg")
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def ROI(img, scale):
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    edges = cv2.Canny(img_blur,30,100)

    # mode – contours를 찾는 방법
    # cv2.RETR_EXTERNAL : contours line중 가장 바같쪽 Line만 찾음.
    # cv2.RETR_LIST : 모든 contours line을 찾지만, hierachy 관계를 구성하지 않음.
    # cv2.RETR_CCOMP : 모든 contours line을 찾으며, hieracy관계는 2-level로 구성함.
    # cv2.RETR_TREE : 모든 contours line을 찾으며, 모든 hieracy관계를 구성함.

    # method –contours를 찾을 때 사용하는 근사치 방법
    # cv2.CHAIN_APPROX_NONE : 모든 contours point를 저장.
    # cv2.CHAIN_APPROX_SIMPLE : contours line을 그릴 수 있는 point 만 저장. (ex; 사각형이면 4개 point)
    # cv2.CHAIN_APPROX_TC89_L1 : contours point를 찾는 algorithm
    # cv2.CHAIN_APPROX_TC89_KCOS : contours point를 찾는 algorithm

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size_min = img.shape[0]

    cnt=0
    pi = 3.14
    pbox = []
    for contr in contours:
        A = cv2.contourArea(contr,False)
        L = cv2.arcLength (contr, False)
        if A>size_min:
            x,y,w,h = cv2.boundingRect(contr)
            if 4*pi*A/(L**2)>0.3:
                cv2.drawContours(img, contr, -1, (0,255,0), 3)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

                pbox.append((x, y, x+w, y+h))

    # print(pbox)
    # print("finally")

                
    return img, pbox

def IoU(box1, box2):
    # box1,2 = (x1, y1, x2, y2) (1은 왼쪽위좌표, 2는 오른쪽 아래좌표)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    # 박스들의 교집합 영역을 구하는 과정 
    # 궁금하면 직접 case분류해서 따라가보면 늘 성립한다는 것을 알게됨
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    # 교집합 계산 완료
    inter = w * h
    # iou = inter / (box1_area + box2_area - inter)
    # 원래는 위에 식이 맞으나 box1_area 즉 욜로의 박스 영역은 그 모양과 크기가 너무 들쑥날쑥해서 비교적 비슷한 크기가
    # 유지되는 box2_area와 inter로만 계산해서 iou를 구했다.
    iou = inter / box2_area

    #print(box2_area)
    #print(iou)
    
    # iou를 리턴
    return iou

def dt(img, pbox):
    (height, width) = (img.shape[0], img.shape[1])
    # Detecting objects
    # blob은 opencv에서 mat타입의 4차원 행렬(N, C, H, W)
    # N 영상 개수 , C 채널 개수 , H 영상 세로 , W 영상 가로
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # print(type(outs)) 튜플 ..
    # print(len(outs))   2...
    # print(outs)
    # print("game over")

    # # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        # print(type(out))   numpy.ndarray
        for detection in out:
            scores = detection[5:]
            # 이 안에 print박아 넣었다고 성능이 너무 후져짐
            # print(type(detection))              numpy.ndarray
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 신뢰도가 0.5가 넘으면 신뢰도 0.95박으니까 예리하게 조져준다.
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 여기까지 만족을 해주어야 인정을 해준다.
                for candidate in range(len(pbox)):
                    # iou가 최소 0.3이 넘도록 세팅
                    if 0.3 < IoU((x, y, x+w, x+h), pbox[candidate]):
                        # print("sdfffffffffffffffffffffff")
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 1)

    return boxes

while True:
    retval, img = cap.read()
    if not retval:
        break

    # 민찬이가 만든 roi박스가 욜로이미지학습에 영향을 끼치지 않도록 깊은 복사로 완전히 새롭게 img_ROI 생성(img와는 완전 별개가 됨)
    img_ROI = copy.deepcopy(img)

    img_ROI, pbox = ROI(img_ROI, scale=1)

    dt(img, pbox)

    # result1 roi영역박스
    # result2 img 욜로와 roi교집합해서 더블체크 후 결과
    cv2.imshow("result1", img_ROI)
    cv2.imshow("result2", img)

    # ## 동영상 녹화
    # out1.write(result)

    key = cv2.waitKey(25)
    if key == 27:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()






