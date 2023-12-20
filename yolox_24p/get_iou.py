import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision

from exp import get_exp
from utils import save_checkpoint
from tqdm import tqdm
from models import Loss_Function
from models import IOUloss
from shapely.geometry import Polygon

from torch.utils.tensorboard import SummaryWriter

count_iter = 0

class Evaluator:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args

        self.num_classes = self.exp.num_classes
        self.start_device = self.args.start_device
        self.numb_device = self.args.devices
        self.device = "cuda" 
        self.loss_func = Loss_Function(self.num_classes)
        self.val_loader = self.exp.get_data_valloader(batch_size=1)
    
        self.input_size = self.exp.input_size
        self.file_name = os.path.join(self.exp.output_dir, self.exp.exp_name)

        os.makedirs(self.file_name, exist_ok=True)

        self.file_list = os.listdir(self.args.load_path)
        
        self.model_weight_path = self.args.weights
        self.prompt_weight_path = self.args.weights_prompt

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

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        
        theta = torch.tensor(15*np.pi/180, device=prediction.device)
        theta_all = torch.arange(24, device=prediction.device) * theta
        cos_theta_all = torch.cos(theta_all)
        sin_theta_all = torch.sin(theta_all)

        box_corner = prediction.new(prediction.shape)

        box_corner[:, :, :26] = prediction[:, :, :26]

        output = [None for _ in range(len(prediction))]
        nms_out_index = torch.tensor([], dtype=torch.long, device=prediction.device)
        
        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                continue
            class_conf, class_pred = torch.max(image_pred[:, 27: 27 + num_classes], 1, keepdim=True)


            conf_mask = (image_pred[:, 26] * class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :27], class_conf, class_pred.float()), 1)
           
            conf_mask_index = torch.nonzero(conf_mask).view(-1)
           
            # detections = detections[conf_mask]
            
            if not detections.size(0):
                continue
            
            cos_theta_all = cos_theta_all.unsqueeze(0).repeat(detections.shape[0], 1)
            sin_theta_all = sin_theta_all.unsqueeze(0).repeat(detections.shape[0], 1)

            p24_x = detections[:, 2:26] * cos_theta_all + detections[:, 0].unsqueeze(1).repeat(1, 24)
            p24_y = detections[:, 2:26] * sin_theta_all + detections[:, 1].unsqueeze(1).repeat(1, 24)


            p24_x_min = p24_x.min(dim=1).values
            p24_x_max = p24_x.max(dim=1).values
            p24_y_min = p24_y.min(dim=1).values
            p24_y_max = p24_y.max(dim=1).values
            
            p24_rect = torch.stack((p24_x_min, p24_y_min, p24_x_max, p24_y_max), dim=0).transpose(0, 1)
            
            nms_out_index = torchvision.ops.nms(
                p24_rect,
                detections[:, 26] * detections[:, 27],
                nms_thre,
            )

            nms_out_list = nms_out_index.tolist()
            conf_mask_list = conf_mask_index.tolist()
            intersection_list = [idx for idx in nms_out_list if idx in conf_mask_list]
            index = torch.tensor(intersection_list, dtype=torch.long)
            
            detections = detections[index] 
            
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output, index
    
    @torch.no_grad()
    def eval(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        use_device = ''
        for gpu in range(self.start_device, self.start_device + self.numb_device):
            use_device = str(gpu) + use_device
        os.environ['CUDA_VISIBLE_DEVICES'] = use_device

        self.model = self.exp.get_model()

        weights_file = torch.load(self.model_weight_path, map_location=self.device)
        # unwanted_keys = ['backbone.norm0.weight', 'backbone.norm0.bias']
        print(weights_file.keys())
        check_backbone = {key.replace('module.', ''): value for key, value in weights_file['model'].items()}
        self.model.load_state_dict(check_backbone)

        total_iou_sum = 0
        total_biou_sum = 0
        total_ciou_sum = 0
        total_matches = 0
        
        prompter = torch.load(self.prompt_weight_path, map_location=self.device)
        prompter.to(self.device)
        prompter.eval()
        
        t = tqdm(total=len(self.val_loader), desc='processing:', ncols=300)
        for step, data in enumerate(self.val_loader):
            images, labels, _, _ = data
            images, labels = images.to(self.device), labels.to(self.device)
            images, labels = self.exp.preprocess(images, labels, self.input_size)
            images, _ = prompter(images)
            
            self.model.to(self.device)
            self.model.eval()
            
            outputs = self.model(images)
            # outputs_ = self.model(images, train=True)
            # outputs_ = outputs_[3]
            outputs, nms_out_index = self.postprocess(outputs, self.num_classes, conf_thre=0.001, nms_thre=0.25)
            

            ious = self.get_iou(outputs[0], labels)
            
            current_iou_sum = sum(iou for _, _, iou, _ , _ in ious)
            current_biou_sum = sum(biou for _, _, _, biou, _ in ious)
            current_ciou_sum = sum(ciou for _, _, _, _, ciou in ious)
            current_matches = len(ious)
            
            if current_matches ==0:
                continue
            
            total_iou_sum += current_iou_sum
            total_biou_sum += current_biou_sum
            total_ciou_sum += current_ciou_sum
            total_matches += current_matches
            
            avg_iou_current = current_iou_sum / current_matches 
            avg_biou_current = current_biou_sum / current_matches
            avg_ciou_current = current_ciou_sum / current_matches
            
            t.set_postfix(avg_iou=avg_iou_current, avg_biou=avg_biou_current, avg_ciou=avg_ciou_current)
            t.update()
        
        avg_iou_total = total_iou_sum / total_matches
        avg_biou_total = total_biou_sum / total_matches
        avg_ciou_total = total_ciou_sum / total_matches   
        t.close()
        print("Total Average IoU:", avg_iou_total)
        print("Total Average BIoU:", avg_biou_total)
        print("Total AVerage CIoU:", avg_ciou_total)
    
    def get_iou(self, output, label):
        scores = output[:, 26] * output[:, 27]
        label = label[:, :, 1:]
        label = label.view(-1, 50)
        torch_pi = torch.tensor(np.pi)
        theta = torch.tensor(15*np.pi/180, device=output.device)
        theta_all = torch.arange(24, device=output.device) * theta
        cos_theta_all = torch.cos(theta_all)
        sin_theta_all = torch.sin(theta_all)
        
        cos_theta_all = cos_theta_all.unsqueeze(0).repeat(output.shape[0],1)
        sin_theta_all = sin_theta_all.unsqueeze(0).repeat(output.shape[0],1)
        
        p24_x = output[:, 2:26].to(torch.int) * cos_theta_all + output[:, 0].unsqueeze(1).repeat(1, 24).to(torch.int)
        p24_y = output[:, 2:26].to(torch.int) * sin_theta_all + output[:, 1].unsqueeze(1).repeat(1, 24).to(torch.int)
        
        
        p24_x_min = p24_x.min(dim=1).values
        p24_x_max = p24_x.max(dim=1).values
        p24_y_min = p24_y.min(dim=1).values
        p24_y_max = p24_y.max(dim=1).values
        
        
        valid_p24_x = (p24_x_min >= 0) & (p24_x_max < 640)
        valid_p24_y = (p24_y_min >= 0) & (p24_y_max < 483)
        
        valid_indices = valid_p24_x & valid_p24_y
        mask = scores >= 0.0001
        
        final_mask = mask & valid_indices
        
        p24_x = p24_x[final_mask]
        p24_y = p24_y[final_mask]
        p_center_x = output[:, 0].to(torch.float)
        p_center_y = output[:, 1].to(torch.float)
        
        g24_x = label[:, 2::2].to(torch.float)
        g24_y = label[:, 3::2].to(torch.float)
        g_center_x = label[:,0].to(torch.float)
        g_center_y = label[:,1].to(torch.float)
        
        g_vect_x = g24_x - g_center_x.reshape((-1, 1))
        g_vect_y = g24_y - g_center_y.reshape((-1, 1))
        g_vect_xy = torch.cat((g_vect_x, g_vect_y), 1).reshape(-1, 2, g_vect_x.shape[1])
        
        p24_x_min = p24_x.min(dim=1).values
        p24_x_max = p24_x.max(dim=1).values
        p24_y_min = p24_y.min(dim=1).values
        p24_y_max = p24_y.max(dim=1).values

        g24_x_min = g24_x.min(dim=1).values
        g24_x_max = g24_x.max(dim=1).values
        g24_y_min = g24_y.min(dim=1).values
        g24_y_max = g24_y.max(dim=1).values
        
        
        
        p24_rect = torch.stack((p24_x_min, p24_y_min, p24_x_max, p24_y_max), dim=0).transpose(0, 1)
        g24_rect = torch.stack((g24_x_min, g24_y_min, g24_x_max, g24_y_max), dim=0).transpose(0, 1)
        
        scale_pd = output[:, 2:26]
        scale_pd = scale_pd[final_mask]
        scale_gt = torch.norm(g_vect_xy, dim=1, out=None, keepdim=False)
        
        # img_with_p = self.draw_points_on_image(image_, p24_x, p24_y)
        # img_with_p_gt = self.draw_points_on_image(image_, g24_x, g24_y)
        # cv2.imshow('Image with Points', img_with_p)
        # cv2.waitKey(0)
        # cv2.imshow('Image with Points', img_with_p_gt)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # ious = self.match_and_calculate_iou(p24_x, p24_y, g24_x, g24_y, p24_rect, g24_rect)
        ious = self.match_and_calculate_ciou(p24_x, p24_y, g24_x, g24_y, g_center_x, g_center_y, scale_gt, p_center_x, p_center_y, scale_pd, p24_rect, g24_rect)
        return ious 
        
    
    def calculate_polygon_iou(self, poly1, poly2):
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return inter_area / union_area
    
    def match_and_calculate_iou(self, p24_x, p24_y, g24_x, g24_y, p24_rect, g24_rect):
        best_matches = []
    
        for i in range(p24_x.shape[0]):
            pred_poly = Polygon(zip(p24_x[i].cpu().numpy(), p24_y[i].cpu().numpy()))
            best_iou = 0
            best_gt_idx = -1

            for j in range(g24_x.shape[0]):
                gt_poly = Polygon(zip(g24_x[j].cpu().numpy(), g24_y[j].cpu().numpy()))
                iou = self.calculate_polygon_iou(pred_poly, gt_poly)

                if iou > best_iou:
                    best_iou = iou
                    box_iou = self.calculate_box_iou(p24_rect[i], g24_rect[j])
                    best_gt_idx = j

            if best_gt_idx != -1:
                best_matches.append((i, best_gt_idx, best_iou, box_iou))

        return best_matches
    
    def match_and_calculate_ciou(self, p24_x, p24_y, g24_x, g24_y, g_center_x, g_center_y, scale_gt, p_center_x, p_center_y, scale_pd, p24_rect, g24_rect):
        best_matches = []
    
        for i in range(p24_x.shape[0]):
            pred_poly = Polygon(zip(p24_x[i].cpu().numpy(), p24_y[i].cpu().numpy()))
            best_iou = 0
            best_gt_idx = -1

            for j in range(g24_x.shape[0]):
                gt_poly = Polygon(zip(g24_x[j].cpu().numpy(), g24_y[j].cpu().numpy()))
                iou = self.calculate_polygon_iou(pred_poly, gt_poly)
                
                if iou > best_iou:
                    best_iou = iou
                    biou = self.calculate_box_iou(p24_rect[i], g24_rect[j])
                    ciou = self.calculate_cir_iou(g_center_x[j], g_center_y[j], scale_gt[j], p_center_x[i], p_center_y[i], scale_pd[i])
                    best_gt_idx = j

            if best_gt_idx != -1:
                best_matches.append((i, best_gt_idx, best_iou, biou, ciou))

        return best_matches
    
    def calculate_cir_iou(self, g_center_x, g_center_y, scale_gt, p_center_x, p_center_y, scale_pd):
        torch_pi = torch.tensor(np.pi)
        g_center_x = g_center_x.unsqueeze(0)
        g_center_y = g_center_y.unsqueeze(0)
        scale_gt = scale_gt.unsqueeze(0)
        p_center_x = p_center_x.unsqueeze(0)
        p_center_y = p_center_y.unsqueeze(0)
        scale_pd = scale_pd.unsqueeze(0)
        
        area_gt_circle = torch_pi * scale_gt**2
        area_pd_circle = torch_pi * scale_pd**2
        
        area_inter, circle_dist = self.circle_inter(g_center_x, g_center_y, scale_gt, p_center_x, p_center_y, scale_pd)
        iou_24 = area_inter / (area_gt_circle + area_pd_circle - area_inter + 1e-6)
        return iou_24.mean().item()
    
    
    def calculate_box_iou(self, box1, box2):
        # 박스 간 교집합 영역 계산
        box1 = box1.unsqueeze(0)  # box1을 2차원 텐서로 변환
        box2 = box2.unsqueeze(0)  # box2도 마찬가지로 2차원 텐서로 변환
        inter_xmin = torch.max(box1[:, 0], box2[:, 0])
        inter_ymin = torch.max(box1[:, 1], box2[:, 1])
        inter_xmax = torch.min(box1[:, 2], box2[:, 2])
        inter_ymax = torch.min(box1[:, 3], box2[:, 3])

        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

        # 박스 간 합집합 영역 계산
        area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area_box1 + area_box2 - inter_area

        # IoU 계산
        iou = inter_area / (union_area + 1e-6)
        return iou.item()
        
    def draw_points_on_image(self, image, p24_x, p24_y):
        # 이미지가 텐서 형식이라면 NumPy 배열로 변환
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)  # [높이, 너비, 채널] 형식으로 변환
            image = (image * 255).astype(np.uint8)  # 스케일링

        # RGB에서 BGR로 변환 (OpenCV는 BGR을 사용)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 각 bounding box에 대한 점들을 찍기
        for i in range(p24_x.shape[0]):
            for j in range(p24_x.shape[1]):
                x, y = int(p24_x[i, j]), int(p24_y[i, j])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 초록색 점

        return image
        
    
    def circle_inter(self, c_gtx, c_gty, gt_r, c_pdx, c_pdy, pd_r):
        torch_pi = torch.tensor(np.pi)
        # 转换格式
        # c_gtx, c_gty：[numb_gt]
        # c_pdx, c_pdy：[numb_pd]
        # gt_r_min :[numb_gt, 24]
        # pd_r_min :[numb_pd, 24]
                
        # 结果缓存 [numb, 24]
        res_inter = torch.zeros_like(gt_r, device = gt_r.device) 

        # 圆心距离 [numb_gt*numb_pd, 1]
        # 所有预测结果和标签结果的距离
        dist = torch.sqrt((c_gtx - c_pdx)**2 + (c_gty - c_pdy)**2)
        dist = dist.unsqueeze(1).repeat(1, 24)
        
        # 正常计算，先找出预测和标签中每个位置中较小的半径[numb_gt*numb_pd, 24]
        if (gt_r.shape[0] == 0) or (pd_r.shape[0] == 0):
            # 如果是占位label，直接返回无效占位数据
            return res_inter,dist
        else:
            min_circle_r, _ = torch.min(torch.stack((gt_r, pd_r), 0), 0)  
            max_circle_r, _ = torch.max(torch.stack((gt_r, pd_r), 0), 0)
             
        ac_min = (min_circle_r**2 + dist**2 - max_circle_r**2)/(2*min_circle_r*dist + 1e-8)
        ac_max = (max_circle_r**2 + dist**2 - min_circle_r**2)/(2*max_circle_r*dist + 1e-8)
        
        ac_min = torch.clip(ac_min, min=-0.99, max=0.99)
        ac_max = torch.clip(ac_max, min=-0.99, max=0.99)

        ang_min = torch.acos(ac_min) # t1
        ang_max = torch.acos(ac_max) # t2

        # 真值和预测结果交集面积[numb_gt*numb_pd, 24] - siou
        inter = ang_min * min_circle_r**2 + ang_max * max_circle_r**2 - min_circle_r * dist * torch.sin(ang_min)       

        # 如果两圆半径差绝对值大于圆心距离，则结果为小圆面积
        min_idx = torch.abs(gt_r - pd_r) >= dist
        min_circle_s = torch_pi * (min_circle_r**2)
        
        # [numb_gt*numb_pd, 24]
        res_inter[min_idx] = min_circle_s[min_idx]

        # 如果圆心大于两半径和,结果为0
        area_0_idx = dist >= gt_r + pd_r
        res_inter[area_0_idx] = 0
        
        # 交集面积赋值
        inter_idx = ~(min_idx + area_0_idx)
        res_inter[inter_idx] = inter[inter_idx]

        # 所有真值和所有预测的面积
        # 面积[numb_gt*numb_pd, 24]
        # 所有真值和所有预测的距离
        # 距离[numb_gt*numb_pd, 24]
        return res_inter, dist
    
    
    
    def tensor_to_cv2(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
        # 이미 NumPy 배열인 경우, 추가 변환 없이 사용
        image = (image).astype(np.uint8)


        # RGB에서 BGR로 변환
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image_bgr

# 이미지에 bounding box를 그리는 함수
    def draw_boxes_on_image(self, image_tensor, pred_boxes, gt_boxes):
        # image_cv2 = self.tensor_to_cv2(image_tensor)
        image_cv2 = image_tensor
        if pred_boxes.nelement() and gt_boxes.nelement() == 0:
            return image_cv2

        # boxes가 단일 박스의 좌표를 포함하는 경우 처리
        if pred_boxes.dim() == 1:
            pred_boxes = pred_boxes.unsqueeze(0)
            gt_boxes = gt_boxes.unsqueeze(0)
            
        for box in pred_boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            cv2.rectangle(image_cv2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Ground truth 박스 그리기 (빨간색)
        for box in gt_boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            cv2.rectangle(image_cv2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        return image_cv2

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")    

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")

    parser.add_argument("-s", "--start_device", default=0, type=int, help="device for start count")
    
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your experiment description file")

    parser.add_argument("-p", "--load_path", type=str, default=None, help="plz input your file path")

    parser.add_argument("-w", "--weights", type=str, default=None, help="plz input your weights path")
    
    parser.add_argument("-w_p", "--weights_prompt", type=str, default=None)

    return parser

@logger.catch
def main(exp, args):
    eval = Evaluator(exp, args)
    eval.eval()

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
        
