"""
读取json文件,生成x,y和24点polygon
"""
import argparse
from numpy.ma import power
import yaml
import os
import json
import cv2

import numpy as np

from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO

class Polygon_24():
    # 모드는 Cord와 Radius이며, Cord는 24개의 점 좌표를 저장하는 것을 의미하고 Radius은 24개의 반지름을 저장하는 것을 의미합니다.
    def __init__(self, mode = "Cord"):
        self.mode = mode
        # 태그 json 파일 경로
        self.json_label_pth = "/media/airlab-jmw/DATA/Dataset/coco2017/annotations/instances_val2017.json"
        # 이미지 파일 경로
        self.image_data_pth = "/media/airlab-jmw/DATA/Dataset/coco2017/images/val2017"
        # 처리된 라벨을 저장하는 경로
        self.new_label_pth = "/media/airlab-jmw/DATA/Dataset/coco2017/labels/val2017_24XY"
        # COCO API 인터페이스
        self.coco = COCO(self.json_label_pth)
        
        self.json_dict = self.load_label_json()
        # 주석 처리 후 dict
        self.label_dict_cord24 = {}
        self.label_dict_radius = {}
        # 레이블 ID에 대한 인덱싱 관계
        self.coco_id2idx = {'1': 0,   '2': 1,   '3': 2,   '4': 3,   '5': 4, 
                            '6': 5,   '7': 6,   '8': 7,   '9': 8,   '10': 9, 
                            '11': 10, '13': 11, '14': 12, '15': 13, '16': 14, 
                            '17': 15, '18': 16, '19': 17, '20': 18, '21': 19, 
                            '22': 20, '23': 21, '24': 22, '25': 23, '27': 24, 
                            '28': 25, '31': 26, '32': 27, '33': 28, '34': 29, 
                            '35': 30, '36': 31, '37': 32, '38': 33, '39': 34, 
                            '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, 
                            '46': 40, '47': 41, '48': 42, '49': 43, '50': 44, 
                            '51': 45, '52': 46, '53': 47, '54': 48, '55': 49, 
                            '56': 50, '57': 51, '58': 52, '59': 53, '60': 54, 
                            '61': 55, '62': 56, '63': 57, '64': 58, '65': 59, 
                            '67': 60, '70': 61, '72': 62, '73': 63, '74': 64, 
                            '75': 65, '76': 66, '77': 67, '78': 68, '79': 69, 
                            '80': 70, '81': 71, '82': 72, '84': 73, '85': 74, 
                            '86': 75, '87': 76, '88': 77, '89': 78, '90': 79}


    # 태그 json 파일 로드
    def load_label_json(self):
        with open(self.json_label_pth,'r') as load_f:
            load_dict = json.load(load_f)
        return load_dict

    # 마스크의 24개 이미지 회전과 수평 축의 교차점을 계산합니다.
    def rotation_for_24p(self, center_x, center_y, mask):
        # 결과 최종 저장용 컨테이너(24점)
        cord_results = []
        # 결과의 최종 저장을 위한 컨테이너(반경 R)
        radius_results = []
        # 이미지의 길이와 너비 가져오기
        img_h, img_w = mask.shape[0], mask.shape[1]
        # 이미지의 대각선 길이(회전선 기준)
        max_line = int(np.sqrt(np.power(img_h,2) + np.power(img_w,2)))
        # 드로잉 템플릿 초기화
        mask_pad = cv2.copyMakeBorder(mask.copy(), max_line, max_line, max_line, max_line,cv2.BORDER_CONSTANT,value= 0)
        # 마스크 정보로 좌표 가져오기
        mask_x, mask_y = np.where(mask_pad != 0)
        # 수평 회전선의 좌표를 초기화합니다.
        horizontal_cord_x = np.arange(0, max_line, 0.2)
        horizontal_cord_y = np.zeros_like(horizontal_cord_x)
        # 좌표 형성
        rot_line = np.array([horizontal_cord_x,
                             horizontal_cord_y])
        for rot_time in range(24):
            # 회전당 하나의 템플릿을 그립니다.
            template = cv2.copyMakeBorder(np.zeros_like(mask), max_line, max_line, max_line, max_line,cv2.BORDER_CONSTANT,value= 0)
            # 한 번에 15도 회전
            theta_rad = rot_time*15*np.pi/180
            # 회전 행렬 정의하기
            M_rot = np.array([[np.cos(theta_rad), -1*np.sin(theta_rad)],
                              [np.sin(theta_rad),    np.cos(theta_rad)]])
            # 회전 결과
            rot_end = np.matmul(M_rot, rot_line).astype(np.int16)
            # 고유성
            rot_end_uniq = rot_end[0,:] + rot_end[1,:]*1j
            _, idx = np.unique(rot_end_uniq, return_index=True)
            rot_end = rot_end[:, idx]
            # 개체의 중앙으로 패닝
            rot_end[0,:] = rot_end[0,:] + center_x + max_line
            rot_end[1,:] = rot_end[1,:] + center_y + max_line
            # 마스크 다시 가로채기 
            template[rot_end[1,:], rot_end[0,:]] = 255
            template[mask_x, mask_y] = 0
            # 광선이 모두 가려지는 것을 방지하려면 추가 픽셀 원이 필요합니다.
            mask_cut = template[max_line - 1:max_line + img_h + 1, max_line - 1:max_line + img_w + 1]
            # 엔드포인트 찾기
            marker_y, marker_x = np.where(mask_cut == 255)
            dist_center = np.sqrt(np.power(marker_x - center_x,2) + np.power(marker_y - center_y,2))
            final_idx = np.argmin(dist_center)
            # 여분의 픽셀 원 안에 있는 마지막 지점은 원본 이미지 크기로 패닝해야 합니다.
            x_final = np.clip(marker_x[final_idx], 0, img_w)
            y_final = np.clip(marker_y[final_idx], 0, img_h)
            # 坐标和半径记录容器
            cord = np.array([x_final, y_final])
            radius = dist_center[final_idx]
            # 좌표 및 반경 기록 컨테이너
            cord_results.append(cord)
            radius_results.append(radius)

        return np.array(cord_results), np.array(radius_results)

    # 对annotations字段进行处理
    # area_t_low 및 area_t_high는 생성된 24포인트 라벨 데이터의 영역과 원본 라벨로 분할된 객체의 영역의 비율에 대한 허용 임계값을 나타냅니다.
    # 24개 지점에서 동그라미로 표시된 개체의 면적이 위의 두 임계값보다 레이블에 표시된 면적보다 작으면 현재 레이블이 삭제됩니다.
    # 532469
    def json_anno_process(self, area_t_low = 0.5, area_t_high = 1.5):        
        # 라벨 정보 가져오기
        anno_info = self.json_dict["annotations"]
        numb_anno = len(anno_info)        

        with tqdm(total=numb_anno) as pbar:
            # 각 라벨 처리
            for anno_numb, anno in enumerate(anno_info):
                cur_image_name = str(anno["image_id"]).zfill(12)
                pbar.set_description('Image_id: {}, anno_numb: {}'.format(cur_image_name, anno_numb))
                # 기존 스토리지 컨테이너에 현재 이미지에 대한 처리 태그 결과가 없는 경우 이미지 항목을 만듭니다.
                if cur_image_name in self.label_dict_cord24:
                    pass
                else:
                    self.label_dict_cord24[cur_image_name] = []

                if cur_image_name in self.label_dict_radius:
                    pass
                else:
                    self.label_dict_radius[cur_image_name] = []
                # 라벨은 클러스터를 버리고 개별 사례만 식별합니다.
                if (not anno["iscrowd"]):
                    # 태그에서 현재 영역의 면적을 읽으며, 면적이 1보다 작은 개체는 처리되지 않습니다.
                    label_area = anno["area"]
                    if (label_area < 1):
                        pbar.update(1)
                        continue
                    # 현재 대상의 카테고리 정보를 검색합니다.
                    class_id = anno["category_id"]
                    # 클래스 및 레이블을 변환해야 합니다.
                    label_id = np.array([self.coco_id2idx[str(class_id)]])
                    # 현재 항목에 대한 정보 처리
                    image_pth = Path(self.image_data_pth) / Path(cur_image_name + ".jpg")
                    # 현재 경로에 파일이 존재하면 이미지를 읽고, 존재하지 않으면 항목을 건너뜁니다.
                    if (os.path.exists(image_pth)):
                        image_ori = cv2.imread(str(image_pth))
                    else:
                        pbar.update(1)
                        continue
                    img_h, img_w = image_ori.shape[0], image_ori.shape[1]
                    img_diag = np.sqrt(np.power(img_h, 2) + np.power(img_w, 2))
                    # 대상의 왼쪽 상단 모서리 좌표를 검색하여 중심 좌표로 변환합니다.
                    obj_x = anno["bbox"][0] + anno["bbox"][2]/2
                    obj_y = anno["bbox"][1] + anno["bbox"][3]/2
                    np.clip(obj_x, 0, img_w)
                    np.clip(obj_y, 0, img_h)
                    # 현재 항목의 마스크 데이터를 계산하여 활성 영역 마스크 값은 1, 나머지는 0입니다.
                    cur_mask = self.coco.annToMask(anno)
                    # 중앙을 기준으로 현재 마스크의 24포인트 가장자리 거리를 계산합니다.
                    cur_24p, cur_24r = self.rotation_for_24p(obj_x, obj_y, cur_mask)
                    # 24시간 거리를 정규화하며, 정규화된 크기는 이미지의 대각선 길이입니다.
                    cur_24r = cur_24r / img_diag
                    # 24개 지점에서 볼록한 베일을 계산하고 베일의 면적을 계산하여 레이블에서 너무 멀리 떨어져 있는 경우 폐기합니다.
                    hull = cv2.convexHull(cur_24p)
                    hull_area = cv2.contourArea(hull)
                    # 비례 설정
                    if (hull_area <= label_area * area_t_low) or (hull_area >= label_area * area_t_high):
                        pbar.update(1)
                        continue
                    else:
                        # 반경 데이터 24개 순서: X축에서 시작하여 시계 방향으로 각각 15° 간격으로 24개
                        obj_cord = np.array([obj_x/img_w, obj_y/img_h])
                        cur_24p = cur_24p.reshape(1, -1).squeeze(0).astype(np.float32)
                        cur_24p[0::2] = cur_24p[0::2]/img_w
                        cur_24p[1::2] = cur_24p[1::2]/img_h
                        label_info_cord24 = np.concatenate((label_id, obj_cord, cur_24p), axis = 0)
                        label_info_radius = np.concatenate((label_id, obj_cord, cur_24r), axis = 0)
                        self.label_dict_cord24[cur_image_name].append(label_info_cord24)  
                        self.label_dict_radius[cur_image_name].append(label_info_radius)
                    # 결과 시각화
                    # self.show_24p(image_ori, hull, cur_24p)
                # 클러스터인 경우 이 현재 태그를 삭제하면 됩니다.
                else:
                    pass

                pbar.update(1)

        return self.label_dict_cord24, self.label_dict_radius

    def show_24p(self, image, hull_points, points_24):
        length = len(hull_points)
        for i in range(len(hull_points)):
            cv2.line(image, tuple(hull_points[i][0]), tuple(hull_points[(i+1)%length][0]), (0,255,0), 2)

        for p in points_24:
            cv2.circle(image, p, 3, (255,0,0), -1)

        cv2.imshow("test1", image)
        cv2.waitKey(0)
    
    # 정보를 텍스트 파일로 저장
    def save_24r_to_txt(self):
        if (self.mode == "Cord"):
            label_dict = self.label_dict_cord24
            # 정보 저장
            format_txt = ["%d"] + ["%0.4f"]*50
        else:
            label_dict = self.label_dict_radius
             # 정보 저장
            format_txt = ["%d"] + ["%0.4f"]*26

        numb_label = len(label_dict)
        with tqdm(total=numb_label) as pbar:   
            for image_numb in label_dict:
                # 텍스트 태그 경로 만들기
                txt_pth = Path(self.new_label_pth) / Path(image_numb + ".txt")
                # 数组转换
                label_info = np.array(label_dict[image_numb])
                # 如果信息不是空
                if label_info.shape[0]:
                    np.savetxt(str(txt_pth), label_info, fmt=format_txt)
                else:
                    np.savetxt(str(txt_pth), label_info)
                pbar.update(1)

if __name__ == "__main__":
    polygon = Polygon_24(mode='')
    polygon.json_anno_process()
    polygon.save_24r_to_txt()