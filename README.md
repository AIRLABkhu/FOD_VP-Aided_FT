# FOD_VP-Aided_FT
 
A guide of Pytorch implementation of "Fisheye Object Detection with Visual Prompting-Aided Finetuning".

### Environment
- Ubuntu 18.04.6 LTS
- Python 3.9.16
- Pytorch 2.0.1

### Dataset
Download WoodScape dataset at 'https://woodscape.valeo.com/woodscape/download'

### Dataset Preprocessing
```bash
$ cd datasets
$ python 2_24_labels_create_wood.py 
```

### Train

```bash
$ python train_24p.py -f load_train/yolox_24p_train.py -b 20 -l 0.0001 
```

### Get 24 point output

```bash
$ python get_iou.py -f load_eval/yolox_24p_eval.py -w {model_weight}.pt -p {validation image path} -w_p {prompt_weight}.pt
```
### Get IOU

```bash
$ python get_iou.py -f load_eval/yolox_24p_eval.py -w {model_weight}.pt -w_p {prompt_weight}.pt
```
