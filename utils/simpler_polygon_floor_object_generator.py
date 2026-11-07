import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import defaultdict
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

class RejectionReason(Enum):
    OCCLUSION = "occlusion"
    OUT_OF_BOUNDS = "out_of_bounds"
    COLLISION = "collision"
    DEPTH_VIOLATION = "depth_violation"

class PolygonFloorObjectGenerator:
    """
    Minimal, unbiased floor object generator.
    - No hardcoded spatial constants
    - No artificial constraints
    - Exact depth ranges
    - Pure random sampling
    """
    
    def __init__(self, categories: List[str], 
                 category_bboxes: Dict[str, np.ndarray],
                 camera,
                 max_retries: int = 50,
                 seed: Optional[int] = None,
                 floor_z: float = 0.0):
        
        self.all_categories = categories
        self.category_bboxes = category_bboxes
        self.floor_z = floor_z
        
        # Calculate volumes for size-based ordering
        self.category_volumes = {}
        for cat in categories:
            if cat in category_bboxes:
                bbox = category_bboxes[cat]
                if not isinstance(bbox, np.ndarray):
                    bbox = np.array(bbox)
                mins = np.min(bbox, axis=0)
                maxs = np.max(bbox, axis=0)
                dims = maxs - mins
                volume = float(np.prod(dims))
                self.category_volumes[cat] = volume
            else:
                self.category_volumes[cat] = 1.0
        
        self.categories = sorted(categories, key=lambda c: self.category_volumes[c])
        
        self.default_max_retries = max_retries
        self.max_retries = max_retries
        
        self._update_camera_parameters(camera)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.visible_polygon = None
        self.prepared_polygon = None
        self.depth_bins = None
        
        self.reset_state()

        # Calculate max footprint radius for each category
        self.category_radii = {}
        for cat in categories:
            if cat in category_bboxes:
                bbox = category_bboxes[cat]
                if not isinstance(bbox, np.ndarray):
                    bbox = np.array(bbox)
                # Max distance from center to any corner (XY plane only)
                center = bbox.mean(axis=0)
                xy_distances = np.linalg.norm(bbox[:, :2] - center[:2], axis=1)
                self.category_radii[cat] = float(np.max(xy_distances))
            else:
                self.category_radii[cat] = 10.0  # Default fallback
    
    def _get_buffered_polygon(self, category: str) -> Optional[Polygon]:
        """Get polygon buffered inward by category's footprint radius."""
        if self.visible_polygon is None:
            return None
        
        radius = self.category_radii.get(category, 10.0)
        
        # Buffer inward (negative value)
        buffered = self.visible_polygon.buffer(-radius)
        
        # Check if buffered polygon is valid and non-empty
        if buffered.is_empty or not buffered.is_valid:
            return self.visible_polygon  # Fallback to original if too small
        
        return buffered

    def _update_camera_parameters(self, camera):
        """Extract camera parameters."""
        self.camera_pos = np.array(camera.location, dtype=float)
        self.look_at = np.array(camera.look_at, dtype=float)
        
        # Camera forward for depth calculation
        self.camera_forward = self.look_at - self.camera_pos
        self.camera_forward = self.camera_forward / np.linalg.norm(self.camera_forward)
        
        # Floor projection
        self.camera_pos_floor = np.array([self.camera_pos[0], self.camera_pos[1]], dtype=float)
        self.look_direction_floor = np.array([self.camera_forward[0], self.camera_forward[1]], dtype=float)
        norm = np.linalg.norm(self.look_direction_floor)
        if norm > 1e-6:
            self.look_direction_floor = self.look_direction_floor / norm
        else:
            self.look_direction_floor = np.array([1.0, 0.0])
    
    def _calculate_actual_depth(self, point_2d: np.ndarray) -> float:
        """Calculate actual depth for floor point."""
        point_3d = np.array([point_2d[0], point_2d[1], self.floor_z])
        return float(np.dot(point_3d - self.camera_pos, self.camera_forward))
    
    def _calculate_depth_range(self):
        """Calculate EXACT depth range - no buffers."""
        if self.visible_polygon is None:
            self.min_depth = 0.0
            self.max_depth = 1000.0
            self.depth_range = 1000.0
            return
        
        coords = list(self.visible_polygon.exterior.coords)
        depths = [self._calculate_actual_depth(np.array(c)) for c in coords]
        
        # EXACT range - no artificial buffers
        self.min_depth = min(depths)
        self.max_depth = max(depths)
        self.depth_range = self.max_depth - self.min_depth
        
        # Calculate polygon dimensions for adaptive parameters
        bounds = self.visible_polygon.bounds
        self.polygon_width = bounds[2] - bounds[0]   # maxx - minx
        self.polygon_height = bounds[3] - bounds[1]  # maxy - miny
    
    def _init_polygon_depth_bins(self, num_bins: int = 50):
        """Create depth bins indexed by actual depth."""
        if self.visible_polygon is None:
            return
        
        bounds = self.visible_polygon.bounds
        minx, miny, maxx, maxy = bounds
        
        # Create evenly-spaced depth bins
        depths = np.linspace(self.min_depth, self.max_depth, num_bins + 1)
        self.depth_bin_edges = depths
        self.depth_bins = [[] for _ in range(num_bins)]
        
        # Sample grid uniformly across polygon
        grid_size = 100
        xs = np.linspace(minx, maxx, grid_size)
        ys = np.linspace(miny, maxy, grid_size)
        
        points_added = 0
        for x in xs:
            for y in ys:
                point = Point(x, y)
                if self.prepared_polygon.contains(point):
                    actual_depth = self._calculate_actual_depth(np.array([x, y]))
                    
                    bin_idx = np.searchsorted(depths, actual_depth) - 1
                    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
                    self.depth_bins[bin_idx].append([x, y])
                    points_added += 1
        
        bin_counts = [len(b) for b in self.depth_bins]
    
    def reset_state(self):
        """Reset state."""
        self.call_count = 0
        self.num_objects_placed = 0
        
        self._rejection_history = []
        self._total_rejections = 0
        self._lifetime_rejections = 0
        self._consecutive_rejections = 0
        self._last_generated_depth = None
        self._last_generated_category = None
        self._last_generated_point = None
        self._last_accepted_depth = -np.inf
        
        self._category_rejection_counts = {cat: 0 for cat in self.categories}
        self._occlusion_count = 0
        self._out_of_bounds_count = 0
        self._collision_count = 0
        self._depth_violation_count = 0
        
        self._bin_usage_count = defaultdict(int)
        self._lateral_attempt_counts = {'edge': 0, 'center': 0}
        self._lateral_success_counts = {'edge': 0, 'center': 0}
        
        # Initialize depth cursor if num_objects is set
        if hasattr(self, 'num_objects'):
            self._advance_depth_cursor()
        else:
            # During __init__, default to min_depth
            self.current_depth = getattr(self, 'min_depth', 0.0)
    
    def _select_category(self, depth: float) -> str:
        """Category selection with size bias."""
        if not self.categories:
            raise ValueError("No categories available")
        
        depth_normalized = (depth - self.min_depth) / max(self.depth_range, 1.0)
        
        weights = []
        total_rejections = sum(self._category_rejection_counts.values())
        
        for i, cat in enumerate(self.categories):
            # Smaller objects closer to camera
            size_factor = 1.0 + (1.0 - depth_normalized) * (1.0 - i / len(self.categories)) * 2.0
            
            # Penalize frequently rejected categories
            if total_rejections > 0:
                rejection_rate = self._category_rejection_counts[cat] / total_rejections
                rejection_penalty = 1.0 / (1.0 + rejection_rate * 5.0)
            else:
                rejection_penalty = 1.0
            
            random_factor = random.uniform(0.8, 1.2)
            weights.append(size_factor * rejection_penalty * random_factor)
        
        rand_val = random.uniform(0.0, sum(weights))
        cumsum = 0.0
        for i, weight in enumerate(weights):
            cumsum += weight
            if rand_val <= cumsum:
                return self.categories[i]
        
        return self.categories[-1]
    
    def _advance_depth_cursor(self):
        """Advance depth cursor with variety for low counts."""
        if self.num_objects <= 10:
            # Low counts: pure random placement
            self.current_depth = random.uniform(self.min_depth, self.max_depth)
        else:
            # High counts: linear progression
            if self.num_objects_placed == 0:
                self.current_depth = self.min_depth
            else:
                progress = self.num_objects_placed / max(self.num_objects - 1, 1)
                self.current_depth = self.min_depth + progress * self.depth_range
    
    def _sample_from_depth_bin(self, target_depth: float, category: str) -> np.ndarray:
        """Sample from depth bin with category-aware bounds checking."""
        if self.depth_bins is None or len(self.depth_bins) == 0:
            return self._uniform_sample_from_polygon(category)
        
        # Get buffered polygon for this category
        safe_polygon = self._get_buffered_polygon(category)
        if safe_polygon is None:
            return self._uniform_sample_from_polygon(category)
        
        prepared_safe = prep(safe_polygon)
        
        # Find target bin
        bin_idx = np.searchsorted(self.depth_bin_edges, target_depth) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.depth_bins) - 1)
        
        # Try nearby bins
        for offset in [0, 1, -1, 2, -2, 3, -3]:
            idx = bin_idx + offset
            if 0 <= idx < len(self.depth_bins) and self.depth_bins[idx]:
                points = self.depth_bins[idx]
                
                # Try multiple points from bin
                for attempt in range(10):
                    point = np.array(random.choice(points))
                    
                    # Check if point is in safe zone
                    if prepared_safe.contains(Point(point[0], point[1])):
                        return point
        
        # Fallback to uniform sampling from safe polygon
        return self._uniform_sample_from_polygon_buffered(safe_polygon)

    def _uniform_sample_from_polygon(self, category: str) -> np.ndarray:
        """Uniform sampling with category-aware buffering."""
        safe_polygon = self._get_buffered_polygon(category)
        if safe_polygon is None:
            safe_polygon = self.visible_polygon
        
        return self._uniform_sample_from_polygon_buffered(safe_polygon)
    
    def _uniform_sample_from_polygon_buffered(self, polygon: Polygon) -> np.ndarray:
        """Uniform fallback sampling from given polygon."""
        if polygon is None:
            return np.array([0.0, 0.0])
        
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds
        prepared = prep(polygon)
        
        for _ in range(1000):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if prepared.contains(Point(x, y)):
                return np.array([x, y])
        
        # Fallback to centroid
        centroid = polygon.centroid
        return np.array([centroid.x, centroid.y])
    
    def report_rejection(self, reason: RejectionReason, 
                        depth: Optional[float] = None, 
                        category: Optional[str] = None,
                        point: Optional[np.ndarray] = None):
        """Report rejection - depth violations don't affect adaptation."""
        depth = depth if depth is not None else self._last_generated_depth
        category = category if category is not None else self._last_generated_category
        
        self._total_rejections += 1
        self._lifetime_rejections += 1
        self._consecutive_rejections += 1
        
        if reason == RejectionReason.OCCLUSION:
            self._occlusion_count += 1
        elif reason == RejectionReason.OUT_OF_BOUNDS:
            self._out_of_bounds_count += 1
        elif reason == RejectionReason.COLLISION:
            self._collision_count += 1
        elif reason == RejectionReason.DEPTH_VIOLATION:
            self._depth_violation_count += 1
            # Don't adapt to depth violations
            return
        
        self._rejection_history.append({
            'depth': depth, 
            'category': category, 
            'reason': reason
        })
        
        if category and reason != RejectionReason.DEPTH_VIOLATION:
            self._category_rejection_counts[category] += 1
    
    def report_acceptance(self, actual_depth: Optional[float] = None,
                         bounds: Optional[np.ndarray] = None,
                         point: Optional[np.ndarray] = None):
        """Report successful placement - no exclusion zones."""
        self._consecutive_rejections = 0
        self._total_rejections = 0
        
        if actual_depth is not None:
            self._last_accepted_depth = actual_depth
        
        # NO occupied_regions tracking - validation handles collisions
        
        # Track lateral success for diagnostics
        if hasattr(self, '_last_attempt_type'):
            self._lateral_success_counts[self._last_attempt_type] += 1
        
        self.num_objects_placed += 1
        self._advance_depth_cursor()
    
    def generate(self, camera) -> Tuple[float, float, str]:
        """Generate next object placement."""
        try:
            depth = self.current_depth
            category = self._select_category(depth)
            
            # Pure random sampling from depth bin
            point_2d = self._sample_from_depth_bin(depth, category)
            
            # Track lateral position (adaptive threshold)
            vec = point_2d - self.camera_pos_floor
            along = np.dot(vec, self.look_direction_floor)
            perp = vec - along * self.look_direction_floor
            lateral_dist = np.linalg.norm(perp)
            
            # Adaptive lateral threshold (10% of polygon width)
            if hasattr(self, 'polygon_width'):
                lateral_threshold = self.polygon_width * 0.1
            else:
                lateral_threshold = 80.0  # Fallback
            
            if lateral_dist < lateral_threshold:
                self._last_attempt_type = 'center'
                self._lateral_attempt_counts['center'] += 1
            else:
                self._last_attempt_type = 'edge'
                self._lateral_attempt_counts['edge'] += 1
            
            self._last_generated_depth = depth
            self._last_generated_category = category
            self._last_generated_point = point_2d
            
            self.call_count += 1
            
            return (float(point_2d[0]), float(point_2d[1]), category)
        
        except Exception as e:
            print(f"Error in generate(): {e}")
            import traceback
            traceback.print_exc()
            
            self.call_count += 1
            fallback_cat = random.choice(self.categories) if self.categories else "default"
            point = self._uniform_sample_from_polygon()
            return float(point[0]), float(point[1]), fallback_cat
    
    def set_num_objects(self, num_objects: int, 
                       camera = None,
                       visible_polygon: Optional[List[Tuple[float, float]]] = None,
                       categories: Optional[List[str]] = None,
                       max_retries: Optional[int] = None):
        """Configure generator for new scene."""
        
        if max_retries is not None:
            self.max_retries = max_retries
        else:
            self.max_retries = self.default_max_retries
        
        if camera is not None:
            self._update_camera_parameters(camera)
        
        if visible_polygon is not None and len(visible_polygon) >= 3:
            self.visible_polygon = Polygon(visible_polygon)
            self.prepared_polygon = prep(self.visible_polygon)
            self._calculate_depth_range()
            self._init_polygon_depth_bins()
        
        if categories is not None:
            valid_categories = [cat for cat in categories if cat in self.all_categories]
            if not valid_categories:
                raise ValueError("No valid categories provided")
            self.categories = sorted(valid_categories, key=lambda c: self.category_volumes[c])
        else:
            self.categories = sorted(self.all_categories, key=lambda c: self.category_volumes[c])
        
        self.num_objects = num_objects
        self.reset_state()
        self.num_objects_placed = 0
    
    def has_exceeded_retries(self) -> bool:
        return self._total_rejections >= self.max_retries
    
    def get_call_count(self) -> int:
        return self.call_count
    
    def get_last_accepted_depth(self) -> float:
        return self._last_accepted_depth
    
    def print_lateral_acceptance_rates(self):
        print("\nLateral Acceptance Analysis:")
        for pos_type in ['edge', 'center']:
            attempts = self._lateral_attempt_counts[pos_type]
            successes = self._lateral_success_counts[pos_type]
            if attempts > 0:
                rate = 100 * successes / attempts
                print(f"  {pos_type.capitalize()}: {successes}/{attempts} = {rate:.1f}% success rate")
    
    def analyze_object_distribution_vs_camera(self, placed_objects):
        """Check if objects cluster around camera look direction."""
        if not placed_objects:
            return
        
        # Get object positions
        positions = np.array([[obj.location[0], obj.location[1]] for obj in placed_objects])
        
        # Project objects onto camera look direction
        look_direction_2d = self.look_direction_floor / np.linalg.norm(self.look_direction_floor)
        
        # For each object, calculate how far it is from the look ray
        lateral_distances = []
        for pos in positions:
            vec_to_obj = pos - self.camera_pos_floor
            # Distance along look direction
            along = np.dot(vec_to_obj, look_direction_2d)
            # Perpendicular distance from look ray
            lateral = np.linalg.norm(vec_to_obj - along * look_direction_2d)
            lateral_distances.append(lateral)
        
        lateral_distances = np.array(lateral_distances)
        
        print(f"\nObject Lateral Spread (from camera look ray):")
        print(f"  Mean distance from center ray: {lateral_distances.mean():.1f}")
        print(f"  Std deviation: {lateral_distances.std():.1f}")
        print(f"  Max distance: {lateral_distances.max():.1f}")
        print(f"  % within 100 units of center: {(lateral_distances < 100).sum() / len(lateral_distances) * 100:.0f}%")
        
        # Also check spatial X spread
        x_coords = positions[:, 0]
        print(f"\nObject Spatial X Distribution:")
        print(f"  X range: [{x_coords.min():.0f}, {x_coords.max():.0f}]")
        print(f"  X std: {x_coords.std():.1f}")

    def get_failure_diagnostics(self) -> Dict:
        return {
            'total_rejections': self._lifetime_rejections,
            'occlusion_count': self._occlusion_count,
            'out_of_bounds_count': self._out_of_bounds_count,
            'collision_count': self._collision_count,
            'depth_violation_count': self._depth_violation_count,
            'objects_placed': self.num_objects_placed,
            'rejection_breakdown': {
                'occlusion': f"{100 * self._occlusion_count / max(self._lifetime_rejections, 1):.1f}%",
                'out_of_bounds': f"{100 * self._out_of_bounds_count / max(self._lifetime_rejections, 1):.1f}%",
                'collision': f"{100 * self._collision_count / max(self._lifetime_rejections, 1):.1f}%",
                'depth_violation': f"{100 * self._depth_violation_count / max(self._lifetime_rejections, 1):.1f}%"
            }
        }