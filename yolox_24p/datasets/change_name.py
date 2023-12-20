import os

directory = '/media/airlab-jmw/DATA/Dataset/instance_annotations'

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        new_name = filename[:5] + '.json'
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        os.rename(old_path, new_path)