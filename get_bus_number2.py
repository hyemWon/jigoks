import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import image_to_string 
plt.style.use('dark_background')


def get_number(filename):
    
    img_ori = cv2.imread(filename)

    image_cropped = img_ori[:210,:440]

    height, width, channel = img_ori.shape

    plt.figure(figsize=(12, 10))
    plt.imshow(image_cropped, cmap='gray')

    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')


    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=23, 
        C=2
    )
    img_thresh=~img_thresh
    plt.figure(figsize=(12, 10))
    plt.imshow(img_thresh, cmap='gray')


    contours, _ = cv2.findContours(
        img_thresh, 
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)

        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2),

        })

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')


    MIN_AREA = 1000

    MIN_WIDTH, MIN_HEIGHT =7, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if MIN_AREA < area  \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
         
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
    #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')


    MAX_DIAG_MULTIPLYER = 4 # 5 
    #컨투어와 컨투어 사이의 길이를 제한
    #측정한 컨투어의 대각선 길이(diag_length)가
    #첫번째 컨투어의 중심과 후보군으로 생각하는 컨투어의 중심 사이의 길이가
    #첫번째 컨투어의 diag_length의 5배 안에 있어야한다고 가정

    MAX_ANGLE_DIFF = 10.0 # 12.0
    #첫번째 컨투어와 두번째 컨투어의 중심을 이었을 때 그릴 수 있는
    #직각삼각형의 사잇각의 최댓값
    #너무 벌어져있으면 번호판 후보군에 들지 않도록

    MAX_AREA_DIFF = 0.5 # 0.5
    #두 컨투어의 면적의 차이
    #면적의 차이가 너무 크게 나면 후보군에 들지 않도록

    MAX_WIDTH_DIFF = 1 #0.8
    #두 컨투어의 너비 차이
    #차이가 너무 크면 번호판 아니라고 인식하도록

    MAX_HEIGHT_DIFF = 0.2 #0.2
    #두 컨투어의 높이 차이

    MIN_N_MATCHED = 3 #3
    #위의 조건들로부터 번호판이라고 예상한 애들이 3개 미만이면
    #번호판이 아니라고 인식하도록
    #3개 이상의 컨투어가 있어야 번호판이라고 인식하도록

    #후보군 찾는 함수
    def find_chars(contour_list): #재귀를 통해 번호판 후보군을 계속 찾음
        matched_result_idx = [] #최종적으로 남는 결과값들의 인덱스를 담아줌
        #print(contour_list)

        for d1 in contour_list:
            matched_contours_idx = []
            matched_contours_idx.append(d1['idx'])
            for d2 in contour_list:
                if d1['idx'] == d2['idx']: #컨투어 두개를 비교하는데 같으면 비교할 필요 없으니까 그냥 넘어감
                    continue

                #d1과 d2의 x와 y의 차이값 (거리의 개념) #['idx']와 ['cx'] 두 개가 별개인지 확인하기!!!!!
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) #d1의 대각길이 

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                #d1과 d2 중심 사이의 거리
                #numpy의 linalg.norm을 사용해서 계산

                if dx == 0: #dx=0이면 중심점의 x좌표가 같은 곳에 다음 컨투어가 있다는 의미 (예를 들면 바로 위쪽)
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
                    matched_contours_idx.append(d2['idx']) #d2만 넣어줌
                #기준에 맞는 애들의 인덱스만 리스트에 넣어줌

            # append this contour
            #matched_contours_idx.append(d1['idx']) #d2만 넣었기 때문에 d1까지 넣어줌

            if len(matched_contours_idx) < MIN_N_MATCHED: #후보군의 길이(개수)가 기준보다 작으면
                continue #번호판에서 제외시켜줌 (번호판일 가능성이 낮기 때문)

            matched_result_idx.append(matched_contours_idx) #기준 개수보다 많으면 최종 후보군에 넣어줌


            ###후보군에 안들어간거
            unmatched_contour_idx = [] #최종 후보군에 오르지 않은 애들도 다시 한번 확인
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx: #후보군에 오른 애들이 아닌 애들을 (후보군 리스트에 없는 애들이면)
                    unmatched_contour_idx.append(d4['idx']) #unmatched_리스트에 넣어줌

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx) #후보군이 아닌 애들을 numpy의 take 활용해서 인덱스 뽑아냄
            #np.take(a, idx) - a에서 idx와 같은 인덱스의 값만 추출

            # recursive
            recursive_contour_list = find_chars(unmatched_contour) #find_chars를 재귀함수로 호출
            #후보군이 아닌 애들 중에서 다시 후보군이 된 녀석들을 담아줌

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)#가능성 있는 후보군들을 가지고 차량번호판이라고 생각되는 리스트 가져옴

    matched_result = []
    for idx_list in result_idx:#차량번호판이라고 생각되는 것을
        matched_result.append(np.take(possible_contours, idx_list)) #최종적으로 matched_result에 넣어줌

  
    temp_result = np.zeros((height, width, channel), dtype=np.uint8) 
    for r in matched_result:
        for d in r:

            #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
            #컨투어에 사각형 그려줌 = (x,y)좌표 찾아서 너비(w)와 높이(h)를 더해줘서 사각형 그려줌

    temp_result=cv2.resize(temp_result, None, None, 2, 2, cv2.INTER_CUBIC)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')



    #삐뚤어져 있는 이미지를 똑바로 돌려줌

    PLATE_WIDTH_PADDING = 1.5 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 1.5
    MAX_PLATE_RATIO = 2

    plate_imgs = []
    plate_infos = []
    plate_trim = []

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




        img_cropped = cv2.getRectSubPix( #번호판만 나오게 이미지를 자름
            img_rotated, #각도를 똑바로 바꿔준 이미지
            patchSize=(int(plate_width), int(plate_height)), #번호판 크기만큼으로 잘라줌
            center=(int(plate_cx), int(plate_cy))
            #center=(int(plate_cx)+int(plate_width)/2, int(plate_cy))
        )
        #앞 번호판에 대한 다음 번호판의 크기 비율이 범위 내에 있으면 추가시켜줌
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
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



        config = ('-l kor+eng --oem 3 --psm 10')
        char = image_to_string(dst, config=config)


        result_chars = ''
        has_digit = False
        for c in char:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit(): #숫자나 한글이 포함되어 있는지
                if c.isdigit(): #숫자가 하나라도 포함되어 있는지
                    has_digit = True
                result_chars += c

        char_list.append(result_chars)

    result = char_list[0]
    return result
    
    
      
        
            
    
