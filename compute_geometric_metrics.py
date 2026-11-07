import torch
import math
import kaolin as kal
from pxr import UsdGeom, Usd
import time 
import os
import argparse
from natsort import natsorted
from tqdm import tqdm
import json

def load_mesh_with_units(usd_path):
    stage = Usd.Stage.Open(usd_path)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    mesh = kal.io.usd.import_mesh(usd_path, triangulate=True)
    mesh.vertices = mesh.vertices * meters_per_unit
    return mesh 

parser = argparse.ArgumentParser()
parser.add_argument('--eval_dataset', default='test_data')
parser.add_argument('--suffix', default='image_2_usd')
parser.add_argument('--percent_eval', type=float, default=0.025)
args = parser.parse_args()

chamfer = 0
f_score = 0
num_samples = 0

max_objects = natsorted(os.listdir(args.eval_dataset))
for num_obj in tqdm(max_objects):
    samples = natsorted(os.listdir(os.path.join(args.eval_dataset, num_obj)))
    num_eval = math.floor(len(samples) * args.percent_eval)
    samples = samples[:num_eval]
    
    mini_chamfer = 0
    mini_fscore = 0
    mini_samples = 0

    for sample in tqdm(samples):
        scene_label = os.path.join(args.eval_dataset, num_obj, sample, 'code.usda')
        scene_test = os.path.join(args.eval_dataset, num_obj, sample, f'pred_{args.suffix}.usda')

        if os.path.exists(scene_label) and os.path.exists(scene_test):
            try:
                mesh_1 = load_mesh_with_units(scene_label)
                mesh_2 = load_mesh_with_units(scene_test)

                verts1 = mesh_1.vertices.unsqueeze(0).cuda()
                verts2 = mesh_2.vertices.unsqueeze(0).cuda()

                p1 = kal.ops.mesh.sample_points(verts1, mesh_1.faces.cuda(), 1000000)[0]
                p2 = kal.ops.mesh.sample_points(verts2, mesh_2.faces.cuda(), 1000000)[0]

                sample_chamfer = kal.metrics.pointcloud.chamfer_distance(p1, p2).item()
                sample_fscore = kal.metrics.pointcloud.f_score(p1, p2, radius=0.2).item()

                vol_data = {
                    "chamfer_dist": sample_chamfer,
                    "fscore": sample_fscore
                }

                with open(os.path.join(args.eval_dataset, num_obj, sample, f'geo_data_{args.suffix}.json'), 'w') as f:
                    json.dump(vol_data, f)

                mini_chamfer += sample_chamfer
                mini_fscore += sample_fscore
                mini_samples += 1
            except Exception as e:
                tqdm.write(f'{e}')
    
    f_score += mini_fscore
    chamfer += mini_chamfer
    num_samples += mini_samples

print(f'F-Score: {f_score / max(1, num_samples)}')
print(f'Chamfer Distance: {chamfer / max(1, num_samples)}')