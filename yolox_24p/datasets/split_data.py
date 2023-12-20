import os
import random
import shutil

random.seed(1)

image_folder = '/media/airlab-jmw/DATA/Dataset/rgb_images'
txt_folder = '/media/airlab-jmw/DATA/Dataset/woodscape_50XY'
json_folder = '/media/airlab-jmw/DATA/Dataset/instance_annotations'

train_images_dir = '/media/airlab-jmw/DATA/Dataset/wood_train/images'
train_anno_dir = '/media/airlab-jmw/DATA/Dataset/wood_train/annotations'
train_json_dir = '/media/airlab-jmw/DATA/Dataset/wood_train/json'

valid_images_dir = '/media/airlab-jmw/DATA/Dataset/wood_valid/images'
valid_anno_dir = '/media/airlab-jmw/DATA/Dataset/wood_valid/annotations'
valid_json_dir = '/media/airlab-jmw/DATA/Dataset/wood_valid/json'

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_anno_dir, exist_ok=True)
os.makedirs(train_json_dir, exist_ok=True)

os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_anno_dir, exist_ok=True)
os.makedirs(valid_json_dir, exist_ok=True)

# Get a list of all file names (without extensions) in the image_folder
filenames = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder)]
# Randomly shuffle the file names
random.shuffle(filenames)

split_point = int(len(filenames)*0.8)
train_filenames = filenames[:split_point]
valid_filenames = filenames[split_point:]

# Copy files into the new directories
for name in train_filenames:
    if os.path.exists(os.path.join(txt_folder, name + '.txt')):
        shutil.copy(os.path.join(image_folder, name + '.png'), os.path.join(train_images_dir, name + '.png'))
        #shutil.copy(os.path.join(txt_folder, name + '.txt'), os.path.join(train_anno_dir, name + '.txt'))
        #shutil.copy(os.path.join(json_folder, name + '.json'), os.path.join(train_json_dir, name + '.json'))
for name in valid_filenames:
    if os.path.exists(os.path.join(txt_folder, name + '.txt')):
        shutil.copy(os.path.join(image_folder, name + '.png'), os.path.join(valid_images_dir, name + '.png'))
        #shutil.copy(os.path.join(txt_folder, name + '.txt'), os.path.join(valid_anno_dir, name + '.txt'))
        #shutil.copy(os.path.join(json_folder, name + '.json'), os.path.join(valid_json_dir, name + '.json'))