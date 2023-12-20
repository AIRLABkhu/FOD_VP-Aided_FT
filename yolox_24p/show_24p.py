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

        logger.info("Evaluating start...")
        # logger.info("\n{}".format(self.model))
       
        with tqdm(total = len(self.file_list), desc='processing:', ncols=70) as t:
            self.model.to(self.device)
            self.model.eval()

            current_time = time.localtime()
            self.save_folder = os.path.join(self.exp.output_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            os.makedirs(self.save_folder, exist_ok=True)   

            for file in self.file_list:
                img_path = os.path.join(self.args.load_path, file)

                eval_img, ratio, ori_image = self.exp.get_data_input(img_path)

                images = eval_img.to(self.device)

                outputs = self.model(images)

                
                outputs, nms_out_index = self.postprocess(outputs, self.num_classes, conf_thre=0.01, nms_thre=0.1)

                self.save_eval_results(ori_image, outputs[0], ratio, file, cls_conf=0.001)    
                
                t.update()
        
        prompter = torch.load(self.prompt_weight_path, map_location=self.device)
        prompter.to(self.device)
        prompter.eval()
        
        avg_iou = 0.0
        avg_box_iou = 0.0
        running_iou = 0.0
        running_box_iou = 0.0
        step_ = 0
        t = tqdm(total=len(self.val_loader), desc='processing:', ncols=200)
        for step, data in enumerate(self.val_loader):
            images, labels, _, _ = data
            images, labels = images.to(self.device), labels.to(self.device)
            images, labels = self.exp.preprocess(images, labels, self.input_size)
            # images, _ = prompter(images)
            
            
            self.model.to(self.device)
            self.model.eval()
            outputs = self.model(images, train=True) 
            outputs_ = self.model(images)

            outputs_, nms_out_index = self.postprocess(outputs_, self.num_classes, conf_thre=0.0, nms_thre=0.25)
            
            loss_all = self.loss_func.forward(outputs, labels, nms_out_index, outputs_[0])

            iou, box_iou, p24_rect, g24_rect = self.loss_func.iou_24
            
            
            if isinstance(box_iou, list) or isinstance(iou, tuple):
                continue
            print(p24_rect)
            print()
            print(g24_rect)
            image = self.tensor_to_cv2(images[0]).copy()
            
            for pred_boxes, gt_boxes in zip(p24_rect, g24_rect):
                # 하나의 이미지에 대한 모든 예측 박스와 ground truth 박스 그리기
                image_with_boxes = self.draw_boxes_on_image(image, pred_boxes, gt_boxes)

                # 이미지 출력
            cv2.imshow('Image with Bounding Boxes', image_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            running_iou += torch.mean(torch.tensor(iou)).item()
            running_box_iou += torch.mean(torch.tensor(box_iou)).item()
            avg_iou = running_iou / (step_+1)
            avg_box_iou = running_box_iou / (step_+1)
            
            t.set_postfix(avg_iou=avg_iou, avg_box_iou = avg_box_iou)
            t.update()
            step_+=1
            
            
            t.close()

        print(avg_iou)
        logger.info(f"average iou : {avg_iou}")
        
    def save_eval_results(self, image, output, ratio, image_name, cls_conf=0.01):
        save_file_name = os.path.join(self.save_folder, os.path.basename(image_name))

        if output is None:
            logger.info("No Detection Results, Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, image)
        else:
            bboxes = output[:, 0:26]

            bboxes /= ratio
            cls = output[:, 28]
            scores = output[:, 26] * output[:, 27]
            mask = cls>4
            cls[mask] = 5
            vis_results = self.vis(image, bboxes, scores, cls, cls_conf, self.COCO_CLASSES)
            
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, vis_results)


    def vis(self, img, boxes, scores, cls_ids, conf, class_names=None):
        theta = 15 * np.pi / 180
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x_center = int(box[0])
            y_center = int(box[1])
            r_start = box[2:].to(torch.int)
            
            if x_center < 0 or x_center >= img.shape[1] or y_center < 0 or y_center >= img.shape[0]:
                continue
            
            color = (self.COLORS[cls_id] * 255).astype(np.uint8).tolist()
            
            text = '{}'.format(class_names[cls_id])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        
            
            
            # 좌표를 저장하는 배열
            points = []
            for idx in range(len(r_start)):
                rot_theta = theta * idx
                cur_x = x_center + r_start[idx] * np.cos(rot_theta)
                cur_y = y_center + r_start[idx] * np.sin(rot_theta)

                if cur_x < 0 or cur_x >= img.shape[1] or cur_y < 0 or cur_y >= img.shape[0]:
                    points = []  # 이미지 밖의 좌표가 발견되면 points 배열을 비워 유효하지 않음을 표시
                    break
                
                points.append((int(cur_x), int(cur_y)))
            
            # points 배열이 비어있지 않다면 모든 좌표가 유효함
            if points:
                for j in range(len(points)):
                    cv2.circle(img, points[j], 2, color, -1)
                    if j > 0:
                        cv2.line(img, points[j], points[j - 1], color, 2)

                # 마지막 점과 시작 점을 연결
                if len(points) > 1:
                    cv2.line(img, points[-1], points[0], color, 2)
                cv2.circle(img, (x_center, y_center), 4, color, -1)
                cv2.putText(img, text, (x_center + 3, y_center - txt_size[1]), font, 0.6, color, thickness=2)
            print(points)
        return img
    
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
        
