import subprocess
import shutil
import functools
import os
import re
from typing import List, Any
import copy
import random
from PIL import Image
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf, Vt
import trimesh
from .asset_directory import address, scale
from .utils_classes import domelight, rectlight, usd_prim, camera, AddCommand

# add a prim to a predefined stage
def add_prim(stage, name, object:usd_prim):
    # define he prim and create it from reference
    prim = stage.DefinePrim(name)
    prim.GetReferences().AddReference(object.address)

    # set type
    type_attr = prim.CreateAttribute("type", Sdf.ValueTypeNames.Token)
    type_attr.Set(object.type)

    # set pose and scale
    UsdGeom.Xformable(prim).ClearXformOpOrder()
    UsdGeom.XformCommonAPI(prim).SetTranslate(object.location)
    UsdGeom.XformCommonAPI(prim).SetRotate(object.rotation, UsdGeom.XformCommonAPI.RotationOrderXYZ)
    UsdGeom.XformCommonAPI(prim).SetScale(object.scale)

    return

def add_rect_light(stage, path, position, rgb, intensity, width=100, height=100):
    """Add rectangular area light (softer shadows)"""
    light = UsdLux.RectLight.Define(stage, path)
    light.CreateColorAttr().Set(Gf.Vec3f(rgb[0], rgb[1], rgb[2]))
    light.CreateIntensityAttr().Set(intensity)
    light.CreateWidthAttr().Set(width)
    light.CreateHeightAttr().Set(height)
    
    xform_api = UsdGeom.XformCommonAPI(light)
    xform_api.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
    
    return

def add_dome_light(stage, path, rgb, intensity, texture_path=None):
    """Add dome (environment) light"""
    light = UsdLux.DomeLight.Define(stage, path)
    light.CreateColorAttr().Set(Gf.Vec3f(rgb[0], rgb[1], rgb[2]))
    light.CreateIntensityAttr().Set(intensity)
    
    # Optional: Use HDRI texture for environment
    if texture_path:
        light.CreateTextureFileAttr().Set(texture_path)
    
    return

# create a USD scene from a set of prims
def create_stage(usd_file_path, prims, light=None):
    stage = Usd.Stage.CreateNew(usd_file_path)
    xform = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(xform)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.centimeters)

    for i, prim in enumerate(prims):
        add_prim(stage, f'/World/prim_{i}', prim)
    
    if light:
        if isinstance(light, domelight):
            add_dome_light(stage, f'/World/stage_light', light.color, light.intensity)
        if isinstance(light, rectlight):
            add_rect_light(stage, f'/World/stage_light', light.location, light.color, light.intensity, light.width, light.height)
    
    stage.GetRootLayer().Save()
    return
 
def encode_number_p10(num):
    sign = '+'
    if num < 0:
        sign = '-'
        num = -num
    num = num + 0.0
    s = '{:.2e}'.format(num)
    exp = str(int(s[5:]) - 2)
    return f'<{sign}><{s[0]}><{s[2]}><{s[3]}><e{exp}>'

def encode_number_p1000(num):
    sign = '+'
    if num < 0:
        sign = '-'
        num = -num
    num = num + 0.0
    s = '{:.2e}'.format(num)
    mantissa = str(int(float(s[:4])*100))
    exp = str(int(s[5:]) - 2)
    return f'<{sign}><{mantissa}><e{exp}>'

def encode_number_b1999(num):
    sign = ''
    if num < 0:
        sign = '-'
        num = -num
    num = num + 0.0
    s = '{:.2e}'.format(num)
    mantissa = str(int(float(s[:4])*100))
    exp = str(int(s[5:]) - 2)
    return f'<{sign}{mantissa}><e{exp}>'

def encode_number_fp15(num):
    sign = ''
    if num < 0:
        sign = '-'
        num = -num
    num = num + 0.0
    s = '{:.2e}'.format(num)
    mantissa = str(int(float(s[:4])*100))
    exp = str(int(s[5:]) - 2)
    return f'<FP{sign}{mantissa}/{exp}>'

def encode_vector(v, encoding_scheme):
    s = ''
    for val in v:
        s += encoding_scheme(val)
    return s

def euler_to_6d(r):
    """
    Convert XYZ Euler angles (intrinsic) to 6D rotation representation.
    
    Args:
        r1, r2, r3: Rotation angles in degrees (X, Y, Z order)
    
    Returns:
        tuple: 6 values representing first two columns of rotation matrix
    """

    # convert angles to radians
    r1, r2, r3 = r[0] * np.pi/180, r[1] * np.pi/180, r[2] * np.pi/180

    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r1), -np.sin(r1)],
        [0, np.sin(r1), np.cos(r1)]
    ])
    
    Ry = np.array([
        [np.cos(r2), 0, np.sin(r2)],
        [0, 1, 0],
        [-np.sin(r2), 0, np.cos(r2)]
    ])
    
    Rz = np.array([
        [np.cos(r3), -np.sin(r3), 0],
        [np.sin(r3), np.cos(r3), 0],
        [0, 0, 1]
    ])
    
    # Intrinsic XYZ: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    
    # Extract first two columns (6 values)
    return (R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1])

def sixd_to_euler(sixd):
    """
    Convert 6D rotation representation to XYZ Euler angles (intrinsic).
    
    Args:
        a1, a2, a3: First column of rotation matrix
        b1, b2, b3: Second column of rotation matrix
    
    Returns:
        tuple: (r1, r2, r3) Euler angles in radians (X, Y, Z order)
    """
    # extract 6D coordinates
    a1, a2, a3, b1, b2, b3 = sixd

    # Reconstruct full rotation matrix
    a = np.array([a1, a2, a3])
    b = np.array([b1, b2, b3])
    
    # Normalize first column
    a = a / np.linalg.norm(a)
    
    # Orthogonalize and normalize second column
    b = b - np.dot(b, a) * a
    b = b / np.linalg.norm(b)
    
    # Compute third column via cross product
    c = np.cross(a, b)
    
    # Construct rotation matrix
    R = np.column_stack([a, b, c])
    
    # Extract Euler angles (XYZ intrinsic order)
    # From rotation matrix to Euler angles
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    
    singular = sy < 1e-6

    if not singular:
        r1 = np.arctan2(R[2,1], R[2,2])  # X rotation
        r2 = np.arctan2(-R[2,0], sy)      # Y rotation
        r3 = np.arctan2(R[1,0], R[0,0])  # Z rotation
    else:
        # Gimbal lock case
        r1 = np.arctan2(-R[1,2], R[1,1])
        r2 = np.arctan2(-R[2,0], sy)
        r3 = 0
    
    return (r1 * 180 / np.pi, r2 * 180 / np.pi, r3 * 180 / np.pi)

def round_vector(vec, precision = 3):
    for i in range(len(vec)):
        vec[i] = round(float(vec[i]), precision)
    return vec

# encode prim according to our method
def encode_prim(prim:usd_prim, encoding_scheme):
    return f'<{prim.type}>' + encode_vector(round_vector(list(prim.location), 3), encoding_scheme) + encode_vector(round_vector(list(euler_to_6d(prim.rotation)), 3), encoding_scheme) + encode_vector(round_vector(list(prim.scale), 3), encoding_scheme)

def encode_camera(camera:camera, encoding_scheme):
    return '<camera>' + encode_vector(round_vector(list(camera.location), 3), encoding_scheme) + encode_vector(round_vector(list(camera.look_at), 3), encoding_scheme)

def encode_domelight(light:domelight, encoding_scheme):
    return '<domelight>' + encode_vector(round_vector(list(light.color), 3), encoding_scheme) + encode_vector([round(light.intensity, 3)], encoding_scheme)

# 3 + 1 + 3 + 1 + 1
def encode_rectlight(light:rectlight, encoding_scheme):
    return '<rectlight>' + encode_vector(round_vector(list(light.color), 3), encoding_scheme) + encode_vector([round(light.intensity, 3)], encoding_scheme) + encode_vector(round_vector(list(light.location), 3), encoding_scheme) + encode_vector([round(light.width, 3)], encoding_scheme) + encode_vector([round(light.height, 3)], encoding_scheme)

num_char = '的'
# encode prim as defined in IG-LLM
def encode_prim_igllm(prim:usd_prim):
    attributes = ['type', 'loc', 'rotation', 'scale']
    attribute_text_val = {'type': prim.type, 'loc': f'({num_char},{num_char},{num_char})', 'rotation': f'({num_char},{num_char},{num_char},{num_char},{num_char},{num_char})', 'scale': f'({num_char},{num_char},{num_char})'}
    attribute_num_vals = {'type': [], 'loc': round_vector(list(prim.location), 3), 'rotation': round_vector(list(euler_to_6d(prim.rotation)), 3), 'scale': round_vector(list(prim.scale), 3)}
    random.shuffle(attributes)
    text = f'add({attributes[0]}={attribute_text_val[attributes[0]]},{attributes[1]}={attribute_text_val[attributes[1]]},{attributes[2]}={attribute_text_val[attributes[2]]},{attributes[3]}={attribute_text_val[attributes[3]]})'
    numbers = []
    for i in range(4):
        numbers += attribute_num_vals[attributes[i]]
    return {'text': text, 'numbers': numbers}

def encode_camera_igllm(cam:camera):
    attributes = ['type', 'loc', 'look_at']
    attribute_text_val = {'type': 'cam', 'loc': f'({num_char},{num_char},{num_char})', 'look_at': f'({num_char},{num_char},{num_char})'}
    attribute_num_val = {'type': [], 'loc': round_vector(list(cam.location), 3), 'look_at': round_vector(list(cam.look_at), 3)}
    random.shuffle(attributes)
    text = f'add({attributes[0]}={attribute_text_val[attributes[0]]},{attributes[1]}={attribute_text_val[attributes[1]]},{attributes[2]}={attribute_text_val[attributes[2]]})'
    numbers = []
    for i in range(3):
        numbers += attribute_num_val[attributes[i]]   
    return {'text': text, 'numbers': numbers}

def encode_domelight_igllm(light:domelight):
    attributes = ['type', 'color', 'intensity']
    attribute_text_val = {'type': 'domelight', 'color': f'({num_char},{num_char},{num_char})', 'intensity': f'{num_char}'}
    attribute_num_val = {'type': [], 'color': round_vector(list(light.color), 3), 'intensity': [round(light.intensity, 3)]}
    random.shuffle(attributes)
    text = f'add({attributes[0]}={attribute_text_val[attributes[0]]},{attributes[1]}={attribute_text_val[attributes[1]]},{attributes[2]}={attribute_text_val[attributes[2]]})'
    numbers = []
    for i in range(3):
        numbers += attribute_num_val[attributes[i]]   
    return {'text': text, 'numbers': numbers}

def encode_rectlight_igllm(light:rectlight):
    attributes = ['type', 'color', 'intensity', 'loc', 'width', 'height']
    attribute_text_val = {'type': 'rectlight', 'color': f'({num_char},{num_char},{num_char})', 'intensity': f'{num_char}', 'loc': f'({num_char},{num_char},{num_char})', 'width': f'{num_char}', 'height': f'{num_char}'}
    attribute_num_val = {'type': [], 'color': round_vector(list(light.color), 3), 'intensity': [round(light.intensity, 3)], 'loc': round_vector(list(light.color), 3), 'width': [round(light.width)], 'height': [round(light.height)]}
    random.shuffle(attributes)
    text = f'add('
    numbers = []
    for i in range(6):
        text += f'{attributes[i]}={attribute_text_val[attributes[i]]}'
        if i != 5:
            text += ','
        numbers += attribute_num_val[attributes[i]]
    text += ')' 
    return {'text': text, 'numbers': numbers}

def encode_prim_igllm_ordered(prim:usd_prim):
    attributes = ['type', 'loc', 'rotation', 'scale']
    attribute_text_val = {'type': prim.type, 'loc': f'({num_char},{num_char},{num_char})', 'rotation': f'({num_char},{num_char},{num_char},{num_char},{num_char},{num_char})', 'scale': f'({num_char},{num_char},{num_char})'}
    attribute_num_vals = {'type': [], 'loc': round_vector(list(prim.location), 3), 'rotation': round_vector(list(euler_to_6d(prim.rotation)), 3), 'scale': round_vector(list(prim.scale), 3)}
    text = f'add({attributes[0]}={attribute_text_val[attributes[0]]},{attributes[1]}={attribute_text_val[attributes[1]]},{attributes[2]}={attribute_text_val[attributes[2]]},{attributes[3]}={attribute_text_val[attributes[3]]})'
    numbers = []
    for i in range(4):
        numbers += attribute_num_vals[attributes[i]]
    return {'text': text, 'numbers': numbers}

def encode_camera_igllm_ordered(cam:camera):
    attributes = ['type', 'loc', 'look_at']
    attribute_text_val = {'type': 'cam', 'loc': f'({num_char},{num_char},{num_char})', 'look_at': f'({num_char},{num_char},{num_char})'}
    attribute_num_val = {'type': [], 'loc': round_vector(list(cam.location), 3), 'look_at': round_vector(list(cam.look_at), 3)}
    text = f'add({attributes[0]}={attribute_text_val[attributes[0]]},{attributes[1]}={attribute_text_val[attributes[1]]},{attributes[2]}={attribute_text_val[attributes[2]]})'
    numbers = []
    for i in range(3):
        numbers += attribute_num_val[attributes[i]]   
    return {'text': text, 'numbers': numbers}

def encode_domelight_igllm_ordered(light:domelight):
    attributes = ['type', 'color', 'intensity']
    attribute_text_val = {'type': 'domelight', 'color': f'({num_char},{num_char},{num_char})', 'intensity': f'{num_char}'}
    attribute_num_val = {'type': [], 'color': round_vector(list(light.color), 3), 'intensity': [round(light.intensity, 3)]}
    text = f'add({attributes[0]}={attribute_text_val[attributes[0]]},{attributes[1]}={attribute_text_val[attributes[1]]},{attributes[2]}={attribute_text_val[attributes[2]]})'
    numbers = []
    for i in range(3):
        numbers += attribute_num_val[attributes[i]]   
    return {'text': text, 'numbers': numbers}

def encode_rectlight_igllm_ordered(light:rectlight):
    attributes = ['type', 'color', 'intensity', 'loc', 'width', 'height']
    attribute_text_val = {'type': 'rectlight', 'color': f'({num_char},{num_char},{num_char})', 'intensity': f'{num_char}', 'loc': f'({num_char},{num_char},{num_char})', 'width': f'{num_char}', 'height': f'{num_char}'}
    attribute_num_val = {'type': [], 'color': round_vector(list(light.color), 3), 'intensity': [round(light.intensity, 3)], 'loc': round_vector(list(light.color), 3), 'width': [round(light.width)], 'height': [round(light.height)]}
    text = f'add('
    numbers = []
    for i in range(6):
        text += f'{attributes[i]}={attribute_text_val[attributes[i]]}'
        if i != 5:
            text += ','
        numbers += attribute_num_val[attributes[i]]
    text += ')' 
    return {'text': text, 'numbers': numbers}

def encode_prim_list_igllm_ordered(prims, light, cam:camera):    
    prims.sort(key = lambda prim: np.dot(cam.look_at - cam.location, np.array(prim.location) - cam.location))
    cam_info = encode_camera_igllm_ordered(cam)
    if isinstance(light, rectlight):
        light_info = encode_rectlight_igllm_ordered(light)
    else:
        light_info = encode_domelight_igllm_ordered(light)
    text = cam_info['text'] + light_info['text']
    numbers = cam_info['numbers'] + light_info['numbers']
    for prim in prims:
        prim_info = encode_prim_igllm_ordered(prim)
        text += prim_info['text']
        numbers += prim_info['numbers']
    return {'text': text, 'numbers': numbers}

def encode_prim_list_igllm_fully_ordered(prims, light, cam:camera):
    def compare_prims(a:usd_prim, b:usd_prim):
        a_vec = np.array(a.location) - cam.location
        b_vec = np.array(b.location) - cam.location
        v_vec = cam.look_at - cam.location
        depth_a = np.dot(a_vec, v_vec)
        depth_b = np.dot(b_vec, v_vec)
        if depth_a != depth_b:
            return depth_a - depth_b
        
        a_look_at_vec = depth_a / np.dot(v_vec, v_vec) * v_vec
        b_look_at_vec = depth_b / np.dot(v_vec, v_vec) * v_vec
        a_lateral = a_vec - a_look_at_vec
        b_lateral = b_vec - b_look_at_vec
        
        up = np.array([0, 0, 1])
        up_vec = up - cam.location
        up_cam = up_vec - (np.dot(up_vec, v_vec) / np.dot(v_vec, v_vec) * v_vec)
        a_up = np.dot(a_lateral, up_cam)
        b_up = np.dot(b_lateral, up_cam)
        if a_up != b_up:
            return a_up - b_up
         
        lateral_cam = np.cross(v_vec, up_cam)
        a_up_perp = np.dot(a_lateral, lateral_cam)
        b_up_perp = np.dot(b_lateral, lateral_cam)
        return a_up_perp - b_up_perp    
    
    prims = sorted(prims, key=functools.cmp_to_key(compare_prims))
    cam_info = encode_camera_igllm_ordered(cam)
    if isinstance(light, rectlight):
        light_info = encode_rectlight_igllm_ordered(light)
    else:
        light_info = encode_domelight_igllm_ordered(light)
    text = cam_info['text'] + light_info['text']
    numbers = cam_info['numbers'] + light_info['numbers']
    for prim in prims:
        prim_info = encode_prim_igllm_ordered(prim)
        text += prim_info['text']
        numbers += prim_info['numbers']
    return {'text': text, 'numbers': numbers}

def encode_prim_list_igllm(prims, light, cam:camera):    
    prims.sort(key = lambda prim: np.dot(cam.look_at - cam.location, np.array(prim.location) - cam.location))
    cam_info = encode_camera_igllm(cam)
    if isinstance(light, rectlight):
        light_info = encode_rectlight_igllm(light)
    else:
        light_info = encode_domelight_igllm(light)
    text = cam_info['text'] + light_info['text']
    numbers = cam_info['numbers'] + light_info['numbers']
    for prim in prims:
        prim_info = encode_prim_igllm(prim)
        text += prim_info['text']
        numbers += prim_info['numbers']
    return {'text': text, 'numbers': numbers}

# make a new encoding where light parameters aren't up for estimation
# just indicate the type of light you want in the scene which will also serve as an eos marker
# domelight = dormlight
# rectlight = rectlight
def encode_prim_list_igllm_without_light(prims, cam:camera):    
    prims.sort(key = lambda prim: np.dot(cam.look_at - cam.location, np.array(prim.location) - cam.location))
    cam_info = encode_camera_igllm(cam)
    text = cam_info['text']
    numbers = cam_info['numbers']
    for prim in prims:
        prim_info = encode_prim_igllm(prim)
        text += prim_info['text']
        numbers += prim_info['numbers']
    return {'text': text + 'end', 'numbers': numbers}

# encode list of prims using our encoding scheme
def encode_prim_list(prims, light, cam:camera, encoding_scheme):
    def compare_prims(a:usd_prim, b:usd_prim):
        a_vec = np.array(a.location) - cam.location
        b_vec = np.array(b.location) - cam.location
        v_vec = cam.look_at - cam.location
        depth_a = np.dot(a_vec, v_vec)
        depth_b = np.dot(b_vec, v_vec)
        if depth_a != depth_b:
            return depth_a - depth_b
        
        a_look_at_vec = depth_a / np.dot(v_vec, v_vec) * v_vec
        b_look_at_vec = depth_b / np.dot(v_vec, v_vec) * v_vec
        a_lateral = a_vec - a_look_at_vec
        b_lateral = b_vec - b_look_at_vec
        
        up = np.array([0, 0, 1])
        up_vec = up - cam.location
        up_cam = up_vec - (np.dot(up_vec, v_vec) / np.dot(v_vec, v_vec) * v_vec)
        a_up = np.dot(a_lateral, up_cam)
        b_up = np.dot(b_lateral, up_cam)
        if a_up != b_up:
            return a_up - b_up
         
        lateral_cam = np.cross(v_vec, up_cam)
        a_up_perp = np.dot(a_lateral, lateral_cam)
        b_up_perp = np.dot(b_lateral, lateral_cam)
        return a_up_perp - b_up_perp
    
    prims = sorted(prims, key=functools.cmp_to_key(compare_prims))
    s = encode_camera(cam, encoding_scheme)
    if isinstance(light, domelight):
        s += encode_domelight(light, encoding_scheme)
    else:
        s += encode_rectlight(light, encoding_scheme)
    for prim in prims:
        s += encode_prim(prim, encoding_scheme)
    return s + '<end_of_file>'

# encode list of prims using our encoding scheme
def encode_prim_list_without_light(prims, cam:camera, encoding_scheme):
    def compare_prims(a:usd_prim, b:usd_prim):
        a_vec = np.array(a.location) - cam.location
        b_vec = np.array(b.location) - cam.location
        v_vec = cam.look_at - cam.location
        depth_a = np.dot(a_vec, v_vec)
        depth_b = np.dot(b_vec, v_vec)
        if depth_a != depth_b:
            return depth_a - depth_b
        
        a_look_at_vec = depth_a / np.dot(v_vec, v_vec) * v_vec
        b_look_at_vec = depth_b / np.dot(v_vec, v_vec) * v_vec
        a_lateral = a_vec - a_look_at_vec
        b_lateral = b_vec - b_look_at_vec
        
        up = np.array([0, 0, 1])
        up_vec = up - cam.location
        up_cam = up_vec - (np.dot(up_vec, v_vec) / np.dot(v_vec, v_vec) * v_vec)
        a_up = np.dot(a_lateral, up_cam)
        b_up = np.dot(b_lateral, up_cam)
        if a_up != b_up:
            return a_up - b_up
         
        lateral_cam = np.cross(v_vec, up_cam)
        a_up_perp = np.dot(a_lateral, lateral_cam)
        b_up_perp = np.dot(b_lateral, lateral_cam)
        return a_up_perp - b_up_perp
    
    prims = sorted(prims, key=functools.cmp_to_key(compare_prims))
    s = encode_camera(cam, encoding_scheme)
    for prim in prims:
        s += encode_prim(prim, encoding_scheme)
    return s + '<end_of_file>'

def encode_prim_list_without_light_shuffled(prims, cam:camera, encoding_scheme):
    random.shuffle(prims)
    s = encode_camera(cam, encoding_scheme)
    for prim in prims:
        s += encode_prim(prim, encoding_scheme)
    return s + '<end_of_file>'
    
def extract_prims_from_usda(usda_string, asset_path = None):
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(usda_string)
    prims = []
    for prim in stage.Traverse():
        attributes = prim.GetAuthoredPropertyNames()
        if 'type' in attributes:
            assert asset_path != None, 'Give asset library path for prim extraction'
            position = prim.GetAttribute('xformOp:translate').Get()
            rotation = prim.GetAttribute('xformOp:rotateXYZ').Get()
            type = prim.GetAttribute('type').Get()
            prim_address = os.path.join(asset_path, address[type])
            prims.append(usd_prim(position, rotation, scale[type], type, prim_address))
            continue
    
    light_prim = stage.GetPrimAtPath('/World/stage_light')
    
    if not light_prim:
        return prims, None
        
    if prim.IsA(UsdLux.DomeLight):
        color = light_prim.GetAttribute('inputs:color').Get()
        intensity = light_prim.GetAttribute('inputs:intensity').Get()
        return prims, domelight(color, intensity)
    
    color = light_prim.GetAttribute('inputs:color').Get()
    intensity = light_prim.GetAttribute('inputs:intensity').Get()
    height = light_prim.GetAttribute('inputs:height').Get()
    width = light_prim.GetAttribute('inputs:width').Get()
    location = light_prim.GetAttribute('xformOp:translate').Get()
    
    return prims, rectlight(color, intensity, location, width, height)

def decode_num(vector, encoding_scheme):
    nums = []
    n = len(vector)
    if encoding_scheme == 'p10':
        for i in range(0, n, 5):
            num = int(vector[i + 1])*100 + int(vector[i + 2])*10 + int(vector[i + 3])
            if vector[i] == '-':
                num = -num
            exp = int(vector[i + 4][1:])
            nums.append(num*(10**exp))
    if encoding_scheme == 'p1000':
        for i in range(0, n, 3):
            num = int(vector[i + 1])
            if vector[i] == '-':
                num = -num
            exp = int(vector[i + 2][1:])
            nums.append(num*(10**exp))
    if encoding_scheme == 'b1999':
        for i in range(0, n, 2):
            num = int(vector[i])
            exp = int(vector[i + 1][1:])
            nums.append(num*(10**exp))
    if encoding_scheme == 'fp15':
        for i in range(n):
            num, exp = vector[i][2:].split('/')
            num, exp = int(num), int(exp)
            nums.append(num*(10**exp))
    return nums

camera_focus = 2
aperature = 2.0955
near_clip = 1
far_clip = 1000000
img_side = 1008

def extract_prims_from_encoding(encoding, encoding_scheme, asset_path = None):
    # extract tokens and exclude empty strings 
    tokens = [s for s in re.split(r'[<>]+', encoding) if s]
    i, n = 0, len(tokens)
    num_length = {}
    num_length['p10'] = 5
    num_length['p1000'] = 3
    num_length['b1999'] = 2
    num_length['fp15'] = 1
    entities = {'cameras': [], 'prims': [], 'lights': []}
    while i < n:
        obj_type = tokens[i]
        if obj_type == 'end_of_file':
            break
        if obj_type == 'camera':
            nums = decode_num(tokens[i + 1: i + 1 + 6*num_length[encoding_scheme]], encoding_scheme)
            location = [nums[0], nums[1], nums[2]]
            look_at = [nums[3], nums[4], nums[5]]
            entities['cameras'].append(camera(location, look_at, camera_focus, aperature, near_clip, far_clip, img_side))
            i = i + 1 + 6*num_length[encoding_scheme]
            continue
        if obj_type == 'rectlight':
            nums = decode_num(tokens[i + 1: i + 1 + 9*num_length[encoding_scheme]], encoding_scheme)
            color = [nums[0], nums[1], nums[2]]
            intensity = nums[3]
            location = [nums[4], nums[5], nums[6]]
            width = nums[7]
            height = nums[8]
            entities['lights'].append(rectlight(color, intensity, location, width, height))
            i = i + 1 + 9*num_length[encoding_scheme]
            continue
        if obj_type == 'domelight':
            nums = decode_num(tokens[i + 1: i + 1 + 4*num_length[encoding_scheme]], encoding_scheme)
            color = [nums[0], nums[1], nums[2]]
            intensity = nums[3]
            entities['lights'].append(domelight(color, intensity))
            i = i + 1 + 4*num_length[encoding_scheme]
            continue
        
        assert asset_path != None, 'Give asset library path for prim extraction'
        prim_address = os.path.join(asset_path, address[obj_type])
        assert os.path.exists(prim_address), f'Asset Library Invalid. Does not contain required assets {address}'
        nums = decode_num(tokens[i + 1: i + 1 + 12*num_length[encoding_scheme]], encoding_scheme)
        position = [nums[0], nums[1], nums[2]]
        rotation = round_vector(list(sixd_to_euler([nums[3], nums[4], nums[5], nums[6], nums[7], nums[8]])), 3)
        scale = [nums[9], nums[10], nums[11]]
        entities['prims'].append(usd_prim(position, rotation, scale, obj_type, prim_address))
        i = i + 1 + 12*num_length[encoding_scheme]
    
    return entities

def extract_prims_from_encoding_without_light(encoding, encoding_scheme, asset_path = None):
    # extract tokens and exclude empty strings 
    tokens = [s for s in re.split(r'[<>]+', encoding) if s]
    i, n = 0, len(tokens)
    num_length = {}
    num_length['p10'] = 5
    num_length['p1000'] = 3
    num_length['b1999'] = 2
    num_length['fp15'] = 1
    entities = {'cameras': [], 'prims': []}
    while i < n:
        obj_type = tokens[i]
        if obj_type == 'end_of_file':
            break
        if obj_type == 'camera':
            nums = decode_num(tokens[i + 1: i + 1 + 6*num_length[encoding_scheme]], encoding_scheme)
            location = [nums[0], nums[1], nums[2]]
            look_at = [nums[3], nums[4], nums[5]]
            entities['cameras'].append(camera(location, look_at, camera_focus, aperature, near_clip, far_clip, img_side))
            i = i + 1 + 6*num_length[encoding_scheme]
            continue
        assert asset_path != None, 'Give asset library path for prim extraction'
        prim_address = os.path.join(asset_path, address[obj_type])
        assert os.path.exists(prim_address), f'Asset Library Invalid. Does not contain required assets {address}'
        nums = decode_num(tokens[i + 1: i + 1 + 12*num_length[encoding_scheme]], encoding_scheme)
        position = [nums[0], nums[1], nums[2]]
        rotation = round_vector(list(sixd_to_euler([nums[3], nums[4], nums[5], nums[6], nums[7], nums[8]])), 3)
        scale = [nums[9], nums[10], nums[11]]
        entities['prims'].append(usd_prim(position, rotation, scale, obj_type, prim_address))
        i = i + 1 + 12*num_length[encoding_scheme]
    return entities

def parse_value(value_str: str) -> Any:
    """Parse attribute value - handles tuples and primitives"""
    value_str = value_str.strip()
    
    # Handle tuples: (1.2, 3, 4)
    if value_str.startswith('(') and value_str.endswith(')'):
        inner = value_str[1:-1]
        elements = []
        for elem in inner.split(','):
            elem = elem.strip()
            if not elem:
                continue
            try:
                elements.append(float(elem) if '.' in elem else int(elem))
            except ValueError:
                elements.append(elem)
        return tuple(elements)
    
    # Handle primitives
    try:
        return float(value_str) if '.' in value_str else int(value_str)
    except ValueError:
        return value_str

def split_by_top_level_comma(content: str) -> List[str]:
    """Split by commas that are not inside parentheses"""
    parts = []
    current = []
    depth = 0
    
    for char in content:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    
    if current:
        parts.append(''.join(current))
    
    return parts

def parse_add_command(command_str: str) -> AddCommand:
    """Parse single add() command string into AddCommand object"""
    
    # Extract content inside add(...)
    match = re.search(r'add\s*\((.*)\)', command_str, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid add command: {command_str}")
    
    content = match.group(1)
    
    # Split by top-level commas (not inside tuples)
    pairs = split_by_top_level_comma(content)
    
    attributes = {}
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        
        # Split on '=' to get key and value
        if '=' not in pair:
            continue
        
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        attributes[key] = parse_value(value)
    
    return AddCommand(**attributes)

def extract_add_commands(text: str) -> List[str]:
    """Extract all add(...) command strings from text"""
    commands = []
    i = 0
    
    while i < len(text):
        # Find 'add('
        match = re.search(r'add\s*\(', text[i:])
        if not match:
            break
        
        start = i + match.start()
        i = i + match.end()
        
        # Find matching closing parenthesis
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            i += 1
        
        commands.append(text[start:i])
    
    return commands

def parse_all_commands(text: str) -> List[AddCommand]:
    """Parse all add() commands from text"""
    command_strings = extract_add_commands(text)
    return [parse_add_command(cmd) for cmd in command_strings]

def extract_prims_from_igllm(encoding, asset_path=None):
    commands = parse_all_commands(encoding)
    entities = {'cameras': [], 'prims': [], 'lights': []}
    for command in commands:
        if command.type == 'cam':
            location = round_vector(list(command.loc), 3)
            look_at = round_vector(list(command.look_at), 3)
            entities['cameras'].append(camera(location, look_at, camera_focus, aperature, near_clip, far_clip, img_side))
            continue
        if command.type == 'domelight':
            color = round_vector(list(command.color), 3)
            intensity = round(command.intensity, 3)
            entities['lights'].append(domelight(color, intensity))
            continue
        if command.type == 'rectlight':
            color = round_vector(list(command.color), 3)
            intensity = round(command.intensity, 3)
            location = round_vector(list(command.loc), 3)
            width = round(command.width, 3)
            height = round(command.height, 3)
            entities['lights'].append(rectlight(color, intensity, location, width, height))
            continue

        assert asset_path != None, 'Give asset library path for prim extraction'
        prim_address = os.path.join(asset_path, address[command.type])
        assert os.path.exists(prim_address), f'Asset Library Invalid. Does not contain required assets {address}'
        position = round_vector(list(command.loc), 3)
        rotation = round_vector(list(sixd_to_euler(command.rotation)), 3)
        scale = round_vector(list(command.scale), 3)
        entities['prims'].append(usd_prim(position, rotation, scale, command.type, prim_address))
    
    return entities

def extract_prims_from_igllm_without_light(encoding, asset_path=None):
    commands = parse_all_commands(encoding)
    entities = {'cameras': [], 'prims': []}
    for command in commands:
        if command.type == 'cam':
            location = round_vector(list(command.loc), 3)
            look_at = round_vector(list(command.look_at), 3)
            entities['cameras'].append(camera(location, look_at, camera_focus, aperature, near_clip, far_clip, img_side))
            continue
        assert asset_path != None, 'Give asset library path for prim extraction'
        prim_address = os.path.join(asset_path, address[command.type])
        assert os.path.exists(prim_address), f'Asset Library Invalid. Does not contain required assets {address}'
        position = round_vector(list(command.loc), 3)
        rotation = round_vector(list(sixd_to_euler(command.rotation)), 3)
        scale = round_vector(list(command.scale), 3)
        entities['prims'].append(usd_prim(position, rotation, scale, command.type, prim_address))
    return entities

def is_valid_igllm_encoding_lightless(encoding):
    """Syntactic validator for a lightless IG-LLM encoding (e.g. the gemma numeric-head
    lightless model output). Returns True iff `encoding` is exactly
        add(<cmd>)add(<cmd>)...add(<cmd>)end
    with no junk before/between/after, with exactly ONE 'cam' command, where every command
    is a well-formed cam or prim (mask each number -> [NUM] and the skeleton must match these
    templates, any attr order):
        cam : attrs == {type=cam,  loc=(N,N,N),  look_at=(N,N,N)}
        prim: attrs == {type=<asset in address>, loc=(N,N,N), scale=(N,N,N), rotation=(N,N,N,N,N,N)}
    Catches what extract_prims_from_igllm_without_light silently mis-parses: wrong numeric
    arity, missing/extra attributes, junk between commands, missing 'end', bad type.
    """
    num_re = re.compile(r'-?\d+(?:\.\d+)?$')

    def num_tuple_ok(value, n):
        value = value.strip()
        if len(value) < 2 or value[0] != '(' or value[-1] != ')':
            return False
        parts = [p.strip() for p in value[1:-1].split(',')]
        return len(parts) == n and all(num_re.match(p) for p in parts)

    s = encoding.strip()
    if not s.endswith('end'):
        return False
    body = s[:-len('end')]

    # body must be exactly add(...)add(...)...  -- balanced parens, nothing in between
    i, commands = 0, []
    while i < len(body):
        if not body.startswith('add(', i):
            return False
        depth, j = 1, i + 4
        while j < len(body) and depth > 0:
            if body[j] == '(':
                depth += 1
            elif body[j] == ')':
                depth -= 1
            j += 1
        if depth != 0:                      # unbalanced / truncated
            return False
        commands.append(body[i:j])
        i = j
    if not commands:
        return False

    cam_count = 0
    for cmd in commands:
        attrs = {}
        for pair in split_by_top_level_comma(cmd[4:-1]):   # strip 'add(' .. ')'
            pair = pair.strip()
            if '=' not in pair:
                return False
            k, v = pair.split('=', 1)
            attrs[k.strip()] = v.strip()
        if attrs.get('type') == 'cam':
            cam_count += 1
            if cam_count > 1:                       # more than one camera -> bail immediately
                return False
            if set(attrs) != {'type', 'loc', 'look_at'}:
                return False
            if not (num_tuple_ok(attrs['loc'], 3) and num_tuple_ok(attrs['look_at'], 3)):
                return False
        else:
            if attrs.get('type') not in address:           # missing/invalid object type
                return False
            if set(attrs) != {'type', 'loc', 'scale', 'rotation'}:
                return False
            if not (num_tuple_ok(attrs['loc'], 3) and num_tuple_ok(attrs['scale'], 3)
                    and num_tuple_ok(attrs['rotation'], 6)):
                return False
    return cam_count == 1   # exactly one camera command required