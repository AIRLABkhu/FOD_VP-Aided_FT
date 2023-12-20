'''
Wood Scape dataset의 .json (semantic segmentation)에서 24 points label를 생성하는 코드
'''
import os
import json
import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class Polygon_24():
    def __init__(self, mode = 'Cord'):
        self.mode = mode
        self.json_label_pth = 'coco_format.json'
        self.image_data_pth = '/media/airlab-jmw/DATA/Dataset/rgb_images'
        self.new_label_pth = '/media/airlab-jmw/DATA/Dataset/woodscape_50XY'
        
        self.coco = COCO(self.json_label_pth)
        self.json_dict = self.load_label_json()

        self.label_dict_cord24 = {}
        self.label_dict_radius = {}
        
        self.coco_id2idx = {'1': 0,   '2': 1,   '3': 2,   '4': 3,   '5': 4,
                            '6': 5,   '7': 6,   '8': 7,   '9': 8,   '10': 9,
                            '11': 10, '12': 11, '13': 12, '14': 13, '15': 14,
                            '16': 15, '17': 16, '18': 17, '19': 18, '20': 19}
        self.COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.0, 0.0, 0.0
            ]
        ).astype(np.float32).reshape(-1, 3)

        self.COCO_CLASSES = (
            "vehicles",
            "person",
            "bicycle",
            "traffic_light",
            "traffic_sign",
            "None"
        )
        
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
    
    def json_anno_process(self, area_t_low = 0.3, area_t_high = 1.3):        
        # 라벨 정보 가져오기
        anno_info = self.json_dict["annotations"]
        numb_anno = len(anno_info)

        with tqdm(total=numb_anno) as pbar:
            cur_image_name_prev = None  # 이전 이미지 이름을 저장하는 변수
            image_ori = None
            all_hull_points = []
            all_points_24 = []
            all_class_names = []
            all_class_colors = []
            
            for anno_numb, anno in enumerate(anno_info):
                cur_image_name = str(anno["image_id"]).zfill(5)
                
                # 새로운 이미지 처리
                if cur_image_name != cur_image_name_prev and cur_image_name_prev is not None:
                    self.show_24p(cur_image_name, image_ori, all_hull_points, all_points_24, all_class_names, all_class_colors)
                    all_hull_points = []
                    all_points_24 = []
                    all_class_names = []
                    all_class_colors = []
                    
                image_pth = Path(self.image_data_pth) / Path(cur_image_name + ".png")
                if os.path.exists(image_pth):
                    image_ori = cv2.imread(str(image_pth))
                else:
                    pbar.update(1)
                    continue

                pbar.set_description('Image_id: {}, anno_numb: {}'.format(cur_image_name, anno_numb))

                if cur_image_name in self.label_dict_cord24:
                    pass
                else:
                    self.label_dict_cord24[cur_image_name] = []

                if cur_image_name in self.label_dict_radius:
                    pass
                else:
                    self.label_dict_radius[cur_image_name] = []

                if not anno["iscrowd"]:
                    label_area = anno["area"]
                    if label_area < 1:
                        pbar.update(1)
                        continue
                    class_id = anno["category_id"]
                    label_id = np.array([self.coco_id2idx[str(class_id)]])

                    class_name = self.COCO_CLASSES[label_id[0]]
                    class_color =  (self.COLORS[label_id[0]] * 255).astype(np.uint8).tolist()
                    
                    img_h, img_w = image_ori.shape[0], image_ori.shape[1]
                    img_diag = np.sqrt(np.power(img_h, 2) + np.power(img_w, 2))
                    
                    obj_x = anno["bbox"][0] + anno["bbox"][2]/2
                    obj_y = anno["bbox"][1] + anno["bbox"][3]/2
                    obj_x = np.clip(obj_x, 0, img_w)
                    obj_y = np.clip(obj_y, 0, img_h)
                    
                    cur_mask = self.coco.annToMask(anno)
                    
                    cur_24p, cur_24r = self.rotation_for_24p(obj_x, obj_y, cur_mask)
                    cur_24p_orgin = cur_24p.copy()
                    
                    cur_24r = cur_24r / img_diag
                    
                    hull = cv2.convexHull(cur_24p)
                    hull_area = cv2.contourArea(hull)
                    
                    if (hull_area <= label_area * area_t_low) or (hull_area >= label_area * area_t_high):
                        pbar.update(1)
                        continue
                    else:
                        obj_cord = np.array([obj_x/img_w, obj_y/img_h])
                        cur_24p = cur_24p.reshape(1, -1).squeeze(0).astype(np.float32)
                        cur_24p[0::2] = cur_24p[0::2]/img_w
                        cur_24p[1::2] = cur_24p[1::2]/img_h
                        label_info_cord24 = np.concatenate((label_id, obj_cord, cur_24p), axis = 0)
                        label_info_radius = np.concatenate((label_id, obj_cord, cur_24r), axis = 0)
                        self.label_dict_cord24[cur_image_name].append(label_info_cord24)  
                        self.label_dict_radius[cur_image_name].append(label_info_radius)

                        all_class_names.append(class_name)
                        all_class_colors.append(class_color)
                    
                        all_hull_points.append(hull)
                        all_points_24.append(cur_24p_orgin)
                        
                cur_image_name_prev = cur_image_name
            
            # 마지막 이미지의 주석을 처리하기 위한 코드
            if image_ori is not None:
                self.show_24p(cur_image_name, image_ori, all_hull_points, all_points_24, all_class_names, all_class_colors)

        return self.label_dict_cord24, self.label_dict_radius

    def show_24p(self,cur_image_name, image, all_hull_points, all_points_24, class_names, class_color):
        
        h, w, c = image.shape
        background = np.ones((h,w,c), dtype=np.uint8)*255
        alpha = 0.5
        image = cv2.addWeighted(image, alpha, background, 1-alpha, 0)
        
        
        for idx, hull_points in enumerate(all_hull_points):
            length = len(hull_points)

            center_x = int(np.mean([pt[0][0] for pt in hull_points]))
            center_y = int(np.mean([pt[0][1] for pt in hull_points])) 

            # for i in range(length):
            #     cv2.line(image, tuple(hull_points[i][0]), tuple(hull_points[(i+1)%length][0]), class_color[idx], 2)

            # cv2.circle(image, (center_x, center_y), 1, class_color[idx], -1)
            # cv2.circle(image, (center_x, center_y), 1, (0,0,255), -1)
            # text_offset_x = 10
            # cv2.putText(image, class_names[idx], (center_x + text_offset_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color[idx], 2)
            
            # for pt in hull_points:
            #     cv2.circle(image, (pt[0][0],pt[0][1]), 3, class_color[idx], -1)
         
            # cv2.imwrite(f'/media/airlab-jmw/DATA/Dataset/gt_images/{cur_image_name}.png', image)
        
        
        for points_24 in all_points_24:
            length = len(points_24)
            for i, p in enumerate(points_24):
                cv2.line(image, (points_24[i][0], points_24[i][1]), (points_24[(i+1)%length][0], points_24[(i+1)%length][1]), (0,255,0), 2)
                cv2.circle(image, (p[0],p[1]), 2, (255,0,0), -1)
                

        cv2.imshow("Annotations", image)
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
    polygon = Polygon_24(mode='Cord')
    polygon.json_anno_process()
    polygon.save_24r_to_txt()