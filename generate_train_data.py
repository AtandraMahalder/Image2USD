import os
import numpy as np
import math
import random
from typing import Optional
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from utils.utils import create_stage
from utils.utils_classes import usd_prim, camera
from utils.simpler_polygon_floor_object_generator import PolygonFloorObjectGenerator, RejectionReason
from utils.asset_directory import categories, scale, address, backgrounds, bounds, length_bounds, min_object_width, shelf_info, obj_gap, fixed_camera, fixed_camera_lookat, environment_catgories, lights
from pxr import Usd, UsdGeom
import json
import argparse
    
# get bounding box
def get_bounding_box(address, scale):
    stage = Usd.Stage.CreateInMemory()
    xform = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(xform)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.centimeters)
    prim = stage.DefinePrim('/World/prim')
    prim.GetReferences().AddReference(address)
    UsdGeom.XformCommonAPI(prim).SetScale(scale)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.ComputeAlignedBox()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    corners = []
    for x in [min_point[0], max_point[0]]:
        for y in [min_point[1], max_point[1]]:
            for z in [min_point[2], max_point[2]]:
                corners.append([x, y, z])
    
    return corners

def rotation_matrix_from_euler(roll, pitch, yaw):
    """Create rotation matrix from Euler angles (in radians)"""
    
    # Roll (rotation around X-axis)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (rotation around Y-axis)  
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (rotation around Z-axis)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def euler_from_rotation_matrix(R):
    """Extract Euler angles (in degrees) from rotation matrix."""
    # Assuming ZYX order (same as construction)
    
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0
    
    return np.degrees(rx), np.degrees(ry), np.degrees(rz)

def create_transformation_matrix(rotation_matrix, translation):
    """Create 4x4 transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

def transform_points_homogeneous(points, transformation_matrix):
    """Transform points using homogeneous coordinates"""
    # Convert to homogeneous coordinates (add 1 as w component)
    points_homo = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_homo = (transformation_matrix @ points_homo.T).T
    
    # Convert back to 3D coordinates
    return transformed_homo[:, :3]

# Usage example
def transform_bbox_full(bbox_corners, rotation_matrix, translation):
    T = create_transformation_matrix(rotation_matrix, translation)
    transformed_corners = transform_points_homogeneous(np.array(bbox_corners), T)
    return transformed_corners.tolist()

def project_points_to_image(world_points, camera_pos, look_at, K):
    # Create camera transformation matrix
    forward = np.array(look_at) - np.array(camera_pos)
    forward = forward / np.linalg.norm(forward)
    
    # Assume up vector is (0, 0, 1) - adjust if different
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # Camera-to-world rotation matrix
    R_cw = np.column_stack([right, up, forward])
    
    # World-to-camera transformation
    R_wc = R_cw.T
    t_wc = -R_wc @ np.array(camera_pos)
    
    projected_points = []
    
    for point in world_points:
        # Transform to camera space
        point_cam = R_wc @ np.array(point) + t_wc
        
        # Skip points behind camera
        if point_cam[2] <= 0:
            continue
            
        # Project to image plane
        point_2d = K @ point_cam
        pixel_x = point_2d[0] / point_2d[2]
        pixel_y = point_2d[1] / point_2d[2]
        
        projected_points.append([pixel_x, pixel_y])
    
    return projected_points

def point_in_polygon_vectorized(points_x, points_y, polygon_vertices):
    """Vectorized ray-casting algorithm (same as before)"""
    n = len(polygon_vertices)
    inside = np.zeros(len(points_x), dtype=bool)
    
    p1x, p1y = polygon_vertices[-1]
    
    for i in range(n):
        p2x, p2y = polygon_vertices[i]
        
        mask = ((points_y > min(p1y, p2y)) & 
                (points_y <= max(p1y, p2y)) & 
                (points_x <= max(p1x, p2x)))
        
        if p1y != p2y:
            xinters = (points_y[mask] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            inside[mask] ^= (points_x[mask] <= xinters)
        
        p1x, p1y = p2x, p2y
    
    return inside

def get_pixels_vectorized(world_points, camera:camera, stride=1):
    """
    Fast vectorized pixel enumeration with optional stride sampling.
    
    Args:
        stride: Sample every Nth pixel (1=all, 2=4x faster, 3=9x faster)
    
    Returns:
        dict with 'pixels' (sampled), 'estimated_total' (scaled estimate)
    """
    camera_pos = camera.location
    look_at = camera.look_at
    K = camera.K
    image_width = camera.img_side
    image_height = camera.img_side
    
    # Project 3D corners to 2D
    pixel_coords = project_points_to_image(world_points, camera_pos, look_at, K)
    
    if len(pixel_coords) < 3:
        return dict()
    
    try:
        # Get bounding box
        xs = [p[0] for p in pixel_coords]
        ys = [p[1] for p in pixel_coords]
        min_x = max(0, int(np.floor(min(xs))))
        max_x = min(image_width - 1, int(np.ceil(max(xs))))
        min_y = max(0, int(np.floor(min(ys))))
        max_y = min(image_height - 1, int(np.ceil(max(ys))))
        
        # Create grid with stride
        x_coords = np.arange(min_x, max_x + 1, stride)
        y_coords = np.arange(min_y, max_y + 1, stride)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Flatten to 1D arrays
        points_x = xx.ravel()
        points_y = yy.ravel()
        
        # Vectorized point-in-polygon test
        inside_mask = point_in_polygon_vectorized(points_x, points_y, pixel_coords)
        
        # Convert back to list of (x, y) tuples (sampled pixels only)
        pixels_inside = [(int(points_x[i]), int(points_y[i])) 
                        for i in range(len(points_x)) if inside_mask[i]]
        
        # Estimate total pixels from sample
        sampled_count = len(pixels_inside)
        estimated_total = sampled_count * (stride * stride)
        
        return {
            'pixels': pixels_inside,           # Actual sampled pixels (for occlusion check)
            'sampled_count': sampled_count,    # How many we found in sample
            'estimated_total': estimated_total, # Scaled-up estimate of true total
            'stride': stride,
            'polygon_coords': pixel_coords,
            'bounds': (min_x, min_y, max_x - min_x, max_y - min_y)
        }
        
    except Exception as e:
        print(f"Error in vectorized pixel computation: {e}")
        return dict()

def normalize(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v

def extract_obb_data(corners):
    """Extract center, axes, and half-extents from 8 corners of OBB"""
    corners = np.array(corners)
    
    # Center is average of all corners
    center = np.mean(corners, axis=0)
    
    # Find three edge vectors from corner 0 to its 3 adjacent corners
    corner_0 = corners[0]
    
    # Find the 3 corners adjacent to corner_0 (closest by distance)
    distances = [np.linalg.norm(corner_0 - corner) for corner in corners[1:]]
    adjacent_indices = np.argsort(distances)[:3] + 1
    
    # Extract the 3 edge vectors
    edge_vectors = []
    for idx in adjacent_indices:
        edge_vectors.append(corners[idx] - corner_0)
    
    # Normalize to get axes and calculate half-extents
    axes = []
    half_extents = []
    
    for edge in edge_vectors:
        length = np.linalg.norm(edge)
        axes.append(normalize(edge))
        half_extents.append(length / 2)
    
    return center, axes, half_extents

def project_obb_onto_axis(corners, axis):
    """Project all corners onto axis and return min/max"""
    projections = [np.dot(corner, axis) for corner in corners]
    return min(projections), max(projections)

def is_separated_on_axis(corners_a, corners_b, axis):
    """Check if two OBBs are separated along given axis"""
    min_a, max_a = project_obb_onto_axis(corners_a, axis)
    min_b, max_b = project_obb_onto_axis(corners_b, axis)
    
    # Separated if no overlap
    return max_a < min_b or max_b < min_a

def obb_intersect(corners_a, corners_b):
    """Check if two oriented bounding boxes intersect using SAT"""
    # Extract OBB data
    center_a, axes_a, _ = extract_obb_data(corners_a)
    center_b, axes_b, _ = extract_obb_data(corners_b)
    
    # Collect all test axes
    test_axes = []
    
    # Add face normals from both boxes
    test_axes.extend(axes_a)
    test_axes.extend(axes_b)
    
    # Add edge-edge cross products
    for axis_a in axes_a:
        for axis_b in axes_b:
            cross = np.cross(axis_a, axis_b)
            if np.linalg.norm(cross) > 1e-6:  # avoid parallel edges
                test_axes.append(normalize(cross))
    
    # Test each axis for separation
    for axis in test_axes:
        if is_separated_on_axis(corners_a, corners_b, axis):
            return False  # Found separating axis
    
    return True  # No separation found = intersection

# grab command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--asset_dir', default='assets')
parser.add_argument('--data_dir', default='usd_data')
parser.add_argument('--max_objects', type=int, default=80)
parser.add_argument('--num_each_count', type=int, default=12)
parser.add_argument('--count_start_index', type=int, default=0)
parser.add_argument('--init_random_seed', type=int, default=0)

args = parser.parse_args()

random.seed(args.init_random_seed)

# resolve asset paths 
asset_path = os.path.realpath(args.asset_dir)
for obj in address.keys():
    address[obj] = os.path.join(asset_path, address[obj])

# create bounding boxes
bounding_box = {}
for cat in categories:
    bounding_box[cat] = np.array(get_bounding_box(address[cat], scale[cat]))


def get_visible_floor_polygon(camera, floor_z, room_bounds):
    """
    Calculate visible floor as trapezoid from near edge to far boundaries.
    """
    cam_pos = camera.location
    K = camera.K
    K_inv = np.linalg.inv(K)
    img_size = camera.img_side
    
    # Sample rays along top and bottom edges of image
    num_samples = 20
    
    # Top edge (y=0)
    top_edge_pixels = np.array([[x, 0, 1] for x in np.linspace(0, img_size-1, num_samples)]).T
    # Bottom edge (y=img_size-1)
    bottom_edge_pixels = np.array([[x, img_size-1, 1] for x in np.linspace(0, img_size-1, num_samples)]).T
    # Left edge (x=0)
    left_edge_pixels = np.array([[0, y, 1] for y in np.linspace(0, img_size-1, num_samples)]).T
    # Right edge (x=img_size-1)
    right_edge_pixels = np.array([[img_size-1, y, 1] for y in np.linspace(0, img_size-1, num_samples)]).T
    
    # Camera orientation
    forward = camera.look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    R_cam_to_world = np.column_stack([right, up, forward])
    
    floor_points = []
    
    # Process all edge rays
    for edge_pixels in [top_edge_pixels, bottom_edge_pixels, left_edge_pixels, right_edge_pixels]:
        rays_camera = K_inv @ edge_pixels
        rays_world = R_cam_to_world @ rays_camera
        
        for i in range(rays_world.shape[1]):
            ray_dir = rays_world[:, i]
            
            # Try floor intersection
            if abs(ray_dir[2]) > 1e-8:
                t_floor = (floor_z - cam_pos[2]) / ray_dir[2]
                
                if t_floor > 0:
                    # Direct floor hit
                    floor_point = cam_pos + t_floor * ray_dir
                else:
                    # Ray points up - intersect with far walls at floor level
                    # then project down to floor
                    far_intersections = []
                    
                    # X boundaries
                    for x_bound in room_bounds['x']:
                        if abs(ray_dir[0]) > 1e-8:
                            t = (x_bound - cam_pos[0]) / ray_dir[0]
                            if t > 0:
                                point = cam_pos + t * ray_dir
                                if (room_bounds['y'][0] <= point[1] <= room_bounds['y'][1]):
                                    # Project to floor
                                    far_intersections.append([x_bound, point[1]])
                    
                    # Y boundaries
                    for y_bound in room_bounds['y']:
                        if abs(ray_dir[1]) > 1e-8:
                            t = (y_bound - cam_pos[1]) / ray_dir[1]
                            if t > 0:
                                point = cam_pos + t * ray_dir
                                if (room_bounds['x'][0] <= point[0] <= room_bounds['x'][1]):
                                    # Project to floor
                                    far_intersections.append([point[0], y_bound])
                    
                    if far_intersections:
                        # Use farthest intersection
                        distances = [np.linalg.norm(np.array(p) - cam_pos[:2]) for p in far_intersections]
                        floor_point = np.array([*far_intersections[np.argmax(distances)], floor_z])
                    else:
                        continue
                
                # Check if point is in room bounds
                if (room_bounds['x'][0] <= floor_point[0] <= room_bounds['x'][1] and
                    room_bounds['y'][0] <= floor_point[1] <= room_bounds['y'][1]):
                    floor_points.append([float(floor_point[0]), float(floor_point[1])])
    
    if len(floor_points) < 3:
        return None, []
    
    # Remove duplicates
    floor_points = np.array(floor_points)
    unique_points = []
    for point in floor_points:
        is_duplicate = False
        for existing in unique_points:
            if np.linalg.norm(point - existing) < 5.0:  # 5 unit tolerance
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    
    if len(unique_points) < 3:
        return None, unique_points
    
    # Calculate convex hull
    from scipy.spatial import ConvexHull
    try:
        polygon_points = np.array(unique_points)
        hull = ConvexHull(polygon_points)
        polygon_points = polygon_points[hull.vertices]
    except:
        polygon_points = np.array(unique_points)
    
    # Bounding box
    min_x, max_x = polygon_points[:, 0].min(), polygon_points[:, 0].max()
    min_y, max_y = polygon_points[:, 1].min(), polygon_points[:, 1].max()
    
    visible_bounds = {
        'x': [min_x, max_x],
        'y': [min_y, max_y],
        'z': room_bounds['z']
    }
    
    return visible_bounds, polygon_points.tolist()

def create_shelves(shelf_type, num_obj, limit):
    pre_flip = shelf_info[shelf_type][0]
    shelf_objects = [usd_prim([0, 0, 0], [0, 0, pre_flip], scale[shelf_type], shelf_type, address[shelf_type])]
    
    num_obj += 1
    if num_obj == limit:
        return shelf_objects
    
    y_shift = shelf_info[shelf_type][1]

    for shelf in shelf_info[shelf_type][2:]:
        z = shelf[0]
        lo_x = shelf[1]
        hi_x = shelf[2]
        objects = shelf[3] + ['skip']

        while hi_x - lo_x >= min_object_width:
            obj_to_place = random.choice(objects)

            if obj_to_place == 'skip':
                lo_x += obj_gap
                continue

            left_pos = random.uniform(lo_x, hi_x)
            obj_pos_x = left_pos - length_bounds[obj_to_place][0]

            if obj_pos_x + length_bounds[obj_to_place][1] > hi_x:
                break

            shelf_objects.append(usd_prim([obj_pos_x, y_shift, z], [0, 0, 0], scale[obj_to_place], obj_to_place, address[obj_to_place]))
            
            num_obj += 1
            if num_obj == limit:
                return shelf_objects

            lo_x = obj_pos_x + length_bounds[obj_to_place][1] + obj_gap

    return shelf_objects

def transform_objects(objects, x, y, z, rx, ry, rz):
    for obj in objects:
        xo, yo, zo = obj.location
        rxo, ryo, rzo = obj.rotation
        xo, yo, zo = x + xo*np.cos(rz) - yo*np.sin(rz), y + xo*np.sin(rz) + yo*np.cos(rz), zo + z
        rxo, ryo, rzo = rxo + rx*180/np.pi, ryo + ry*180/np.pi, rzo + rz*180/np.pi
        obj.location = [xo, yo, zo]
        obj.rotation = [rxo, ryo, rzo]
    return objects

def shelf_rot(x, y, cam):
    target_x = cam.location[0] - x
    target_y = cam.location[1] - y
    if target_x != 0:
        theta_target = np.arctan(target_y/target_x)
    elif target_y > 0:
        theta_target = np.pi/2
    else:
        theta_target = -np.pi/2
    if target_x < 0:
         theta_target += np.pi
    
    theta = theta_target + np.pi/2
    
    return theta*180/np.pi

def create_objects(loc, rot, obj_type, num_obj, limit):
    if obj_type in shelf_info.keys():
        objects = create_shelves(obj_type, num_obj, limit)
        objects = transform_objects(objects, loc[0], loc[1], loc[2], rot[0]*np.pi/180, rot[1]*np.pi/180, rot[2]*np.pi/180)
        return objects
    
    objects = [usd_prim(loc, rot, scale[obj_type], obj_type, address[obj_type])]
    return objects

def sample_light_parameters(mode: Optional[str] = None):
    """Uniform sampling across color and intensity space"""
    r = random.uniform(0.5, 1.0)
    g = random.uniform(0.5, 1.0)
    b = random.uniform(0.5, 1.0)
    intensity = random.uniform(5000, 10000)
    return (r, g, b), intensity

max_placement_tries = 500
visibility_threshold = 90
camera_focus = 2
aperature = 2.0955
near_clip = 1
far_clip = 1000000
img_side = 1008

generator = PolygonFloorObjectGenerator(
    categories=categories,
    category_bboxes=bounding_box,
    camera=camera((0, 0, 0), (1, 1, 1), camera_focus, aperature, near_clip, far_clip, img_side),
    max_retries=max_placement_tries,
    seed=args.init_random_seed,
    floor_z=0
)

def bound_and_pixel_validity(prims, P, camera, last_accepted_depth):
    pixels = set()
    bounds_calc = False
    obj_bounds = []
    global empty_dict
    global zero_pixel
    for prim in prims:
        bounds = bounding_box[prim.type]
        transformed_bounds = transform_bbox_full(bounds, 
                                                 rotation_matrix_from_euler(prim.rotation[0]*np.pi/180, 
                                                                            prim.rotation[1]*np.pi/180, 
                                                                            prim.rotation[2]*np.pi/180),
                                                prim.location)
        if bounds_calc == False:
            obj_bounds = transformed_bounds
            bounds_calc = True

            # Calculate actual depth of object
            forward = (camera.look_at - camera.location) / np.linalg.norm(camera.look_at - camera.location)
            actual_depth = np.dot(np.array(prim.location) - camera.location, forward)

            # Validate depth ordering
            if actual_depth < last_accepted_depth:
                return RejectionReason.DEPTH_VIOLATION, set(), obj_bounds, actual_depth

        pix = get_pixels_vectorized(transformed_bounds, camera)

        if not pix:
            return RejectionReason.OUT_OF_BOUNDS, set(), obj_bounds, actual_depth
        
        pix = set(pix['pixels'])
        if len(pix) == 0:
            return RejectionReason.OUT_OF_BOUNDS, set(), obj_bounds, actual_depth

        if len(pix & P) < math.ceil(visibility_threshold / 100 * len(pix)):
            return RejectionReason.OCCLUSION, pix, obj_bounds, actual_depth

        pixels = pixels | pix

    return 'accept', pixels, obj_bounds, actual_depth

def create_scene(n:int, background, cam):
    visible_bounds, floor_points = get_visible_floor_polygon(cam, bounds[background]['z'][0], bounds[background])
    
    # initialize pixel set
    pixels = set((i, j) for i in range(1008) for j in range(1008))

    global generator
    generator.set_num_objects(n, cam, floor_points, environment_catgories[background], max_placement_tries)

    num_obj = 0
    objs = [usd_prim((0, 0, 0), (0, 0, 0), scale[background], background, address[background])]
    last_accepted_depth = float('-inf')
    accepted_bounds = []

    while num_obj < n:
        while not generator.has_exceeded_retries():
            x, y, obj_type = generator.generate(cam)

            # print(f'{x} {y} {obj_type}')

            if obj_type in shelf_info.keys():
                rot = shelf_rot(x, y, cam) + random.uniform(-45, 45)
                new_objs = create_objects([x, y, bounds[background]['z'][0]], [0, 0, rot], obj_type, num_obj, n)
                # print(f'Create {obj_type} with rotation {rot}')
            else:
                new_objs = create_objects([x, y, bounds[background]['z'][0]], [0, 0, random.uniform(0, 360)], obj_type, num_obj, n)

            pix_valid, O, obj_bounds, actual_depth = bound_and_pixel_validity(new_objs, pixels, cam, last_accepted_depth)
        
            if pix_valid == RejectionReason.OCCLUSION:
                generator.report_rejection(RejectionReason.OCCLUSION)
                # print('Rejected due to occlusion')
                continue

            if pix_valid == RejectionReason.OUT_OF_BOUNDS:
                generator.report_rejection(RejectionReason.OUT_OF_BOUNDS)
                # print('Rejected due to out of bounds')
                continue

            if pix_valid == RejectionReason.DEPTH_VIOLATION:
                generator.report_rejection(RejectionReason.DEPTH_VIOLATION)
                # print('Rejected due to depth violation')
                continue

            collision = False
            for box in accepted_bounds:
                if obb_intersect(box, obj_bounds):
                    collision = True
                    break
            
            if collision:
                generator.report_rejection(RejectionReason.COLLISION)
                # print('Rejected due to collision')
                continue

            # print(f'Placed {num_obj + len(new_objs)} objects after {generator._total_rejections} tries')
            generator.report_acceptance(actual_depth)
            num_obj += len(new_objs)
            objs = objs + new_objs
            pixels.difference_update(O)
            last_accepted_depth = actual_depth
            accepted_bounds.append(obj_bounds)
            break

        if generator.has_exceeded_retries():
            # print(f'Retry the entire scene again after having placed {num_obj} objects')
            generator.set_num_objects(n, cam, floor_points, environment_catgories[background], max_placement_tries)
            objs = [usd_prim((0, 0, 0), (0, 0, 0), scale[background], background, address[background])]
            num_obj = 0
            pixels = set((i, j) for i in range(1008) for j in range(1008))
            last_accepted_depth = float('-inf')
            accepted_bounds = []
    
    color, intensity = sample_light_parameters()
    return objs, color, intensity

os.makedirs(args.data_dir, exist_ok=True)

# main generation loop
id = 0
for num_obj in tqdm(range(1, args.max_objects + 1)):
    os.makedirs(os.path.join(args.data_dir, f'{num_obj}'), exist_ok=True)
    for idx in tqdm(range(args.count_start_index, args.count_start_index + args.num_each_count)):
        background = backgrounds[id % 3]
        camera_loc = fixed_camera[background][(id // 3) % 4]
        camera_look_at = fixed_camera_lookat[background]
        cam = camera(camera_loc, camera_look_at, camera_focus, aperature, near_clip, far_clip, img_side)
        objs, light_color, light_intensity = create_scene(num_obj, background, cam)
        lighting = lights[background]
        lighting.color = light_color
        lighting.intensity = light_intensity
        create_stage(os.path.join(args.data_dir, f'{num_obj}', f'sample-{idx}', 'code.usda'), objs, lighting)
        
        camera_data = {
            'location': list(camera_loc),  # Convert numpy array to list
            'look_at': list(camera_look_at)
        }
        with open(os.path.join(args.data_dir, f'{num_obj}', f'sample-{idx}', 'camera.json'), 'w') as f:
            json.dump(camera_data, f)

        id += 1