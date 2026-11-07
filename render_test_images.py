# fix import issues
import os
import sys
import time

sim_startup_start = time.time()
from isaacsim import SimulationApp
simulation_app = SimulationApp(launch_config={"headless": True, "renderer": "PathTracedLighting",})
import carb.settings
import omni.replicator.core as rep
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import open_stage
import omni.usd
from PIL import Image
import numpy as np
import json
from pxr import Usd, UsdGeom, Sdf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='test_data')
parser.add_argument('--max_num_obj', type=int, default=80)
parser.add_argument('--suffix', default='image_2_usd')
parser.add_argument('--start_sample', type=int, default=0)
parser.add_argument('--end_sample', type=int, default=119)
args = parser.parse_args()

def add_payload(stage, prim_path, usd_file_path):
    """Add USD payload to stage"""
    prim = stage.DefinePrim(prim_path)
    prim.GetPayloads().AddPayload(usd_file_path)
    return prim

def remove_payload(stage, prim_path):
    """Remove payload and prim"""
    stage.RemovePrim(prim_path)

# Enable scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
carb.settings.get_settings().set("/app/asyncRendering", False)

# Setup the environment
assets_root_path = get_assets_root_path()
stage_utils.create_new_stage()
world = World()
stage = omni.usd.get_context().get_stage()

cam = rep.create.camera(focal_length=20, horizontal_aperture = 20.995)

rp = rep.create.render_product(cam, resolution=(1008, 1008))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([rp])

prim_path = "/World/CurrentObject" 

for num_obj in range(1, args.max_num_obj + 1):
    for i in range(args.start_sample, args.end_sample + 1):
        # add file to stage
        if not os.path.exists(os.path.join(args.data_dir, f'{num_obj}', f'sample-{i}', f'pred_{args.suffix}.usda')):
            continue
        
        add_payload(stage, prim_path, os.path.join(args.data_dir, f'{num_obj}', f'sample-{i}', f'pred_{args.suffix}.usda'))

        # move camera
        with open(os.path.join(args.data_dir, f'{num_obj}', f'sample-{i}', f'pred_camera_{args.suffix}.json'), 'r') as f:
            cam_info = json.load(f)

        with cam:
            rep.modify.pose(
                position = cam_info['location'], 
                look_at = cam_info['look_at']
            )
        
        # update stage
        for _ in range(30):
            simulation_app.update()
        world.step(render=True)

        # take image
        rep.orchestrator.step()
        rgb_data = rgb_annotator.get_data()

        # save image
        Image.fromarray(rgb_data).save(os.path.join(args.data_dir, f'{num_obj}', f'sample-{i}', f'pred_{args.suffix}.png'))

        # remove file
        remove_payload(stage, prim_path)

        # update stage
        for _ in range(30):
            simulation_app.update()
        world.step(render=True)

simulation_app.close()