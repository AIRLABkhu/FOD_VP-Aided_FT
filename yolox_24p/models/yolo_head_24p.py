import math
from loguru import logger

import cv2
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import bboxes_iou, postprocess

from .network_blocks import BaseConv, DWConv

class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        # num_class에서 바이너리 코드 아래의 비트 수를 계산합니다.
        self.num_classes = num_classes
        
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # 카테고리별 예측 결과, 80개 카테고리
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # 위치 예측 결과, x,y,24*r, 총 26개
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=26,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # 예상 배경 예측 결과, 1
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, train=False): # xin[0].shape=> [batch, 192, 80, 80] / xin[1].shape=> [batch, 384, 40, 40] / 
                                         # xin[2].shape=> [batch, 768, 20, 20]
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
                          
            if train:
                # 최종 출력 메시지 합성
                # 출력 크기: [배치, 107, 80/40/20, 80/40/20] 34 -> 26 + 1 + 80 
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                                
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                
                x_shifts.append(grid[:, :, 0]) # grid [1, 6400, 2]
                y_shifts.append(grid[:, :, 1])

                expanded_strides.append(torch.zeros(1, grid.shape[1], device='cuda').fill_(stride_this_level)) # [1, 6400]

                # 초기 단계에서는 l1 사용 안 함, 끝으로 갈수록 켜짐
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 26, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 26
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                # output = torch.cat([reg_output, obj_output, cls_output], 1)

            outputs.append(output)

        if train:
            return x_shifts, y_shifts, expanded_strides, torch.cat(outputs, 1), origin_preds
        
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            
            # 进该项（True）
            if self.decode_in_inference: 
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 27 + self.num_classes
        
        # 피처 맵 크기, 각각 80, 40, 20
        hsize, wsize = output.shape[-2:]
        
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)]) # yv , xv shape: [80,80]/[40,40]/[20,20]
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid # [1, 1, 80, 80, 2]
        # output - [16(batch), 107, 80, 80]
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        ) # output = [16(batch), 6400, 107]
        grid = grid.view(1, -1, 2) # grid [1, 6400, 2]
    
        # 중심점 위치 예측
        output[..., :2] = (output[..., :2] + grid) * stride
        # 중심점 위치 예측 나머지 위치 길이 예측
        output[..., 2:26] = torch.exp(output[..., 2:26]) * stride

        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # 8400개의 각 결과를 그리드 데이터에 추가하고 사진 결과를 복구한다.
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:26] = torch.exp(outputs[..., 2:26]) * strides

        return outputs








                

    
