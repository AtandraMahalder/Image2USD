import os
import math
import shutil
import argparse
import random
import json
from tqdm import tqdm

# grab command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--asset_dir', default='assets')
parser.add_argument('--data_dir', default='usd_data')
parser.add_argument('--prepared_data', default='train_data')
parser.add_argument('--max_objects', type=int, default=80)
parser.add_argument('--num_each_object', type=int, default=1200)
args = parser.parse_args()

asset_dir = os.path.realpath(args.asset_dir)
os.makedirs(f'{args.prepared_data}', exist_ok=True)
splits = ['train', 'validation']
idx = {}
data = {}
for split in splits:
    os.makedirs(os.path.join(f'{args.prepared_data}', split), exist_ok=True)
    idx[split] = 1
    data[split] = []

random.seed(0)

for num in tqdm(range(1, args.max_objects + 1)):
    for scene_type in range(12):
        indices = set([i for i in range(args.num_each_object) if i % 12 == scene_type])
        split_indices = {}

        num_samples = len(indices)
        num_train = math.ceil(0.9*num_samples)
        split_indices['train'] = random.sample(list(indices), num_train)
        for i  in split_indices['train']:
            indices.discard(i)
        
        split_indices['validation'] = list(indices)

        for split in splits:
            for i in tqdm(split_indices[split], desc=f'Creating data of type {scene_type} for {split} from {num} objects'):
                source_image_path = os.path.join(args.data_dir, f'{num}', f'sample-{i}', 'image.png')
                dest_image_path = os.path.join(f'{args.prepared_data}', split, f'sample-{idx[split]}.png')
                shutil.copyfile(source_image_path, dest_image_path)

                usd_string = ''
                with open(os.path.join(args.data_dir, f'{num}', f'sample-{i}', 'code.usda'), 'r') as file:
                    usd_string = file.read()
                
                with open(os.path.join(args.data_dir, f'{num}', f'sample-{i}', 'camera.json'), 'r') as file:
                    cam_info = json.load(file)
                
                data[split].append({'file_name': f'sample-{idx[split]}.png', 'text': usd_string, 'cam_location': cam_info['location'], 'cam_look_at': cam_info['look_at']})
                idx[split] += 1

for split in splits:
    with open(os.path.join(f'{args.prepared_data}', split, 'metadata.jsonl'), 'w') as file:
        for obj in tqdm(data[split], desc=f'creating metadata file for {split}'):
            file.write(json.dumps(obj) + '\n')