# -*- coding: utf-8 -*-
import numpy as np

#조건으로 걸러주기!!!
MAX_DIAG_MULTIPLYER = 5 # 5  대각선길이
MAX_ANGLE_DIFF = 12.0 # 12.0  1번째 contour와 2번째 contour 의 각도
MAX_AREA_DIFF = 0.5 # 0.5  면적의 차이
MAX_WIDTH_DIFF = 0.8 # 너비 차이
MAX_HEIGHT_DIFF = 0.2 # 높이 차이
MIN_N_MATCHED = 3 # 3 # 위에 조건들이 3개이상 충족해야 번호판이다

def find_chars(contour_list):
    matched_result_idx = []
    
    
# 이중for문으로 예를들면 첫번째 contour와 두번째 contour를 비교

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            
# np.linalg.norm(a - b) 벡터 a와 벡터 b 사이의 거리를 구한다.
# 삼각함수 사용
            
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) #면적의 비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w'] # 너비의 비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h'] # 높이의 비율

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        # 번호판 후보군의 윤곽선 개수가 3보다 작으면 번호판일 확률이 낮다. 이유는 한국 번호판은 총 7자리 이기 때문이다.
        matched_contours_idx.append(d1['idx'])
        # 윤곽선 개수가 3보다 작을때는 continue를 통해 제외
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue
        # 최종 후보군에 넣어주기
        matched_result_idx.append(matched_contours_idx)
        # 아닌 것들을 다시 한번 비교하고 넣어준다.
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(contour_list, unmatched_contour_idx)  
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
