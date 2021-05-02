#자동차 번호 인식, tesseract 활용

import get_bus_number2

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt #이미지 프로세싱 결과 나타내기 위함
import pytesseract
from pytesseract import image_to_string
plt.style.use('dark_background')

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import auth

        
# default_app = firebase_admin.initialize_app()


cred = credentials.Certificate('C:/Users/lenovo/jigoks-2421b-firebase-adminsdk-h3pr1-7ee5c0c1d5.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://jigoks-2421b-default-rtdb.firebaseio.com/'
})  

filename = sys.argv[1]

# 이미지 파일 이름
img_ori = cv2.imread(filename)
height, width, channel = img_ori.shape #이미지 사이즈 저장

dst = cv2.fastNlMeansDenoisingColored(img_ori,None,10,10,7,21)
img_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

##hsv에서 yellow 범위
lower_yellow = (10,30,30)
upper_yellow = (25, 255, 255)

img_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow) ##노란색 추출할 범위 지정

img_result = cv2.bitwise_and(dst, dst, mask = img_mask)

plt.figure(figsize=(12, 10)) 
plt.imshow(img_result)



# Convert Image to Grayscale
gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY) #그레이컬러로 변환

plt.figure(figsize=(12, 10))
plt.imshow(gray, cmap='gray')


# Maximize Contrast
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) #(3,3)크기의 사각형 모양의 커널을 만들어줌

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray_new = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)



# Adaptive Thresholding
img_blurred = cv2.GaussianBlur(gray, ksize=(3,3), sigmaX=0) #노이즈 줄이기 위해 사용(배경의 노이즈를 제거해줌)

#검은색, 흰색 두가지의 색으로만 이미지가 나타나도록
img_thresh = cv2.adaptiveThreshold( #이미지에 threshold(기준값)지정해서 기준보다 낮으면 0, 높으면 1로( binary로 값 바꿈)
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)

contours, _ = cv2.findContours( #이미지에서 윤곽선을 찾음
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))


plt.figure(figsize=(12, 10))
plt.imshow(temp_result)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

#list에 컨투어 정보들 저장
contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)#컨투어의 사각형 범위 찾아냄
    
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    

    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        #컨투어 감싼 사각형의 중심좌표들도 저장
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')


MIN_AREA = 20
MIN_WIDTH, MIN_HEIGHT = 1,2
MIN_RATIO, MAX_RATIO = 0.25, 2 #가로대비 세로비율의 최소,최대 값 0.25, 1.0 

possible_contours = [] #번호판일 가능성이 있는 컨투어들의 값을 리스트에 저장

cnt = 0
for d in contours_dict: #이전에 저장한 컨투어 array를 사용
    area = d['w'] * d['h'] #넓이
    ratio = d['w'] / d['h'] #가로대비세로 비율
    
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt #저장할 때 인덱스도 같이 저장해줌
        cnt += 1
        possible_contours.append(d)
        
# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
#     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')


MAX_DIAG_MULTIPLYER = 2.7 # 5 
MAX_ANGLE_DIFF = 7.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 1 #0.8
MAX_HEIGHT_DIFF = 0.2 #0.2
MIN_N_MATCHED = 3 #3


#후보군 찾는 함수
def find_chars(contour_list): #재귀를 통해 번호판 후보군을 계속 찾음
    matched_result_idx = [] #최종적으로 남는 결과값들의 인덱스를 담아줌
    #print(contour_list)
    
    for d1 in contour_list:
        matched_contours_idx = []
        matched_contours_idx.append(d1['idx'])
        for d2 in contour_list:
            if d1['idx'] == d2['idx']: 
                continue
            
           
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) #d1의 대각길이 
            
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
           
            
            if dx == 0: 
                angle_diff = 90 #dx=0이면 0으로 나누는 계산이 안되기 때문에 따로 정해줌
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
                #아크탄젠트로 사잇각인 라디안을 구해서 degrees를 통해 각도로 변환
                
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) #면적 비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w'] #가로 비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h'] #세로 비율
            #면적, 너비, 높이의 비율을 구해줌
            
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx']) 
            
       
        if len(matched_contours_idx) < MIN_N_MATCHED: 
            continue #번호판에서 제외시켜줌 (번호판일 가능성이 낮기 때문)

        matched_result_idx.append(matched_contours_idx) 

        
        ###후보군에 안들어간거
        unmatched_contour_idx = [] #최종 후보군에 오르지 않은 애들도 다시 한번 확인
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx']) #unmatched_리스트에 넣어줌

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx) 
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)#가능성 있는 후보군들을 가지고 차량번호판이라고 생각되는 리스트 가져옴

matched_result = []
for idx_list in result_idx:#차량번호판이라고 생각되는 것을
    matched_result.append(np.take(possible_contours, idx_list)) #최종적으로 matched_result에 넣어줌

#print(matched_result)
# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8) 
for r in matched_result:
    for d in r:
        
        #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        #컨투어에 사각형 그려줌 = (x,y)좌표 찾아서 너비(w)와 높이(h)를 더해줘서 사각형 그려줌

temp_result=cv2.resize(temp_result, None, None, 2, 2, cv2.INTER_CUBIC)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')



#find_chars를 통해 나온 result_idx들을 모아서 그려줌

#삐뚤어져 있는 이미지를 똑바로 돌려줌

PLATE_WIDTH_PADDING = 5 # 1.3
PLATE_HEIGHT_PADDING = 3 # 1.5
MIN_PLATE_RATIO = 1.5
MAX_PLATE_RATIO = 2

plate_imgs = []
plate_infos = []
char_list = []

for i, matched_chars in enumerate(matched_result):#최종 result에 대해 loop를 돌면서
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) #리스트의 원소들을 순차적으로 정렬
    
    #번호판이라고 생각되는 컨투어 상자들의 모음을 통해 번호판의 센터(x,y)좌표를 계산
    #sorted_chars[0] : 번호판의 맨 처음 (맨 처음 컨투어 상자)
    #sorted_chars[-1] : 번호판의 맨 끝(맨 마지막 컨투어 상자)
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2 #맨앞 중심좌표와 맨끝 중심좌표의 평균을 구해서 중심좌표 구함
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    #번호판 너비
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    #번호판의 끝부분 x좌표에서 너비 w를 더해주고 맨앞 x좌표를 빼주면 너비가 나옴
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h'] #높이를 다 더해줌

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING) #높이 평균 구해줌
    #너비와 높이까지 구해줌
    #번호판의 기울기로 만들어진 삼각형의 사이 각도를 구함
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy'] #삼각형의 높이
    #맨 끝의 y중심값에서 맨 앞 y중심값을 빼줌
    
    triangle_hypotenus = np.linalg.norm( #빗변의 길이
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )#첫번째 번호판의 센터 좌표와 끝 번호판의 센터 좌표 차이
    #norm으로 거리를 구함
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) #높이/빗변과 아크사인함수로 각도 구함
    #라디안 값을 '도'로 바꿔줌
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    #로테이션매트릭스를 구함
    
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    #삐뚤어진 이미지를 똑바로 돌림(각도만큼 회전시켜줌)
    
    #print(plate_width, plate_height)
    #print(plate_cx, plate_cy)
    
    img_cropped = cv2.getRectSubPix( #번호판만 나오게 이미지를 자름
        img_rotated, #각도를 똑바로 바꿔준 이미지
        patchSize=(int(plate_width), int(plate_height)), #번호판 크기만큼으로 잘라줌
        center=(int(plate_cx), int(plate_cy))
    )
    #앞 번호판에 대한 다음 번호판의 크기 비율이 범위 내에 있으면 추가시켜줌
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        print(img_cropped.shape[1])
        continue #범위 벗어나면 continue로 for문으로 다시 돌아감
    
    plate_imgs.append(img_cropped) # 번호판 전체의 이미지가 나타나도록 넣어줌
    plate_infos.append({ #번호판의 (x,y)좌표 값, (w,h)값
        'x': int(plate_cx - plate_width / 2), #중심위치에서 너비/2 값을 빼면 x좌표값 나옴
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    
     # 이미지 자르기    
    for i in plate_infos:

        x, y, w, h = int(i['x']), int(i['y']), int(i['w']), int(i['h'])
        img_trim = img_ori[y:y+h, x:x+w]

        _,image_result = cv2.threshold(img_trim, 127, 255, cv2.THRESH_BINARY)


        sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 9, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 9.0

        dst = cv2.filter2D(image_result, -1, sharpening_2)


        config = ('-l kor+eng --oem 3 --psm 11')
        char = image_to_string(dst, config=config)


        result_chars = ''
        has_digit = False
        for c in char:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit(): #숫자나 한글이 포함되어 있는지
                if c.isdigit(): #숫자가 하나라도 포함되어 있는지
                    has_digit = True
                result_chars += c


        char_list.append(result_chars)
        
        plt.figure(figsize=(5, 4))
        plt.imshow(dst)





#=================================================================

#70~79
result = 0

if (char_list[0][4] in ['아','바','사','자']) and (70<=int(char_list[0][2:4])<=79):
 
    result = get_bus_number2.get_number(filename)

print(result)


    
#===================================================================
          

dir = db.reference()
dir.update({'bus': result})