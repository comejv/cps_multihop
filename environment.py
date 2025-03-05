from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np

env_logger = getLogger("environment")


class ManhattanGrid:
    """Manhattan grid environment (3x3 grid, 1km x 1km)"""

    def __init__(self, grid_size=3, block_length=250, road_width=16):
        self.grid_size = grid_size
        self.block_length = block_length
        self.total_size = grid_size * block_length * 2  # Accounts for road width
        self.road_width = road_width
        self.roads = self._create_roads()
        self.intersections = self._create_intersections()
        self.obstacles = self._place_obstacles()
        self.los_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Add vehicle dimensions
        self.vehicle_width = 2.0  # meters
        self.vehicle_length = 5.0  # meters

        # Add road segment mapping for efficiency
        self.road_segments = self._create_road_segments()

        env_logger.info(
            f"Initialized Manhattan Grid: {grid_size}x{grid_size}, {len(self.obstacles)} obstacles"
        )
        env_logger.debug(f"Road width: {road_width}m, block length: {block_length}m")

    def _create_roads(self):
        """Create horizontal and vertical roads"""
        roads = []
        # Create road positions (center of roads)
        for i in range(self.grid_size + 1):
            pos = i * self.block_length * 2
            # Horizontal road
            roads.append(
                {"start": (0, pos), "end": (self.total_size, pos), "type": "horizontal"}
            )
            # Vertical road
            roads.append(
                {"start": (pos, 0), "end": (pos, self.total_size), "type": "vertical"}
            )
        return roads

    def _create_road_segments(self):
        """Create a mapping of road segments for efficient vehicle LOS checking"""
        segments = {}

        for i, road in enumerate(self.roads):
            road_id = i
            road_type = road["type"]
            start, end = road["start"], road["end"]

            # Create road segment identifier
            if road_type == "horizontal":
                y = start[1]  # y-coordinate is constant for horizontal roads
                segments[road_id] = {
                    "type": "horizontal",
                    "y": y,
                    "x_min": min(start[0], end[0]),
                    "x_max": max(start[0], end[0]),
                }
            else:  # vertical
                x = start[0]  # x-coordinate is constant for vertical roads
                segments[road_id] = {
                    "type": "vertical",
                    "x": x,
                    "y_min": min(start[1], end[1]),
                    "y_max": max(start[1], end[1]),
                }

        return segments

    def _create_intersections(self):
        """Identify intersections of roads"""
        intersections = []
        for h_road in [r for r in self.roads if r["type"] == "horizontal"]:
            for v_road in [r for r in self.roads if r["type"] == "vertical"]:
                y = h_road["start"][1]
                x = v_road["start"][0]
                intersections.append((x, y))
        return intersections

    def _place_obstacles(self):
        """Place buildings in the city blocks to create realistic urban environment"""
        obstacles = []

        # Find road positions for creating blocks
        horizontal_roads = [
            r["start"][1] for r in self.roads if r["type"] == "horizontal"
        ]
        vertical_roads = [r["start"][0] for r in self.roads if r["type"] == "vertical"]

        # Sort road positions
        horizontal_roads.sort()
        vertical_roads.sort()

        # Create buildings in each city block
        for i in range(len(horizontal_roads) - 1):
            for j in range(len(vertical_roads) - 1):
                # Calculate block boundaries
                x1 = (
                    vertical_roads[j] + self.road_width / 2 + 20
                )  # Add margin from road
                y1 = horizontal_roads[i] + self.road_width / 2 + 20
                x2 = vertical_roads[j + 1] - self.road_width / 2 - 20
                y2 = horizontal_roads[i + 1] - self.road_width / 2 - 20

                # Create one large building per block
                # This ensures communication across perpendicular roads is blocked
                obstacles.append((x1, y1, x2, y2))

        env_logger.debug(f"Created {len(obstacles)} buildings in city blocks")

        return obstacles

    def is_in_line_of_sight(
        self, v1: "Vehicle", v2: "Vehicle", all_vehicles=None, wireless=False
    ):
        """Check if two positions have line of sight
        (not blocked by obstacles or other vehicles if not wireless)"""

        # Check if vehicles are on the same road and lane for caching
        same_lane = (
            self._check_on_same_road(v1, v2) and v1.lane_offset == v2.lane_offset
        )

        # Use cache for same-lane vehicles
        if same_lane and not wireless:
            cache_key = (v1.id, v2.id)

            # Check if we have a cache entry
            if cache_key in self.los_cache:
                result, (v1_pos, v2_pos) = self.los_cache[cache_key]

                env_logger.debug(
                    f"Using cached LOS result for vehicles {v1.id} and {v2.id}: {result}"
                )
                self.cache_hits += 1
                return result
            else:
                self.cache_misses += 1

        # Proceed with original LOS calculation if no cache hit
        pos1 = v1.position
        pos2 = v2.position
        x1, y1 = v1.position
        x2, y2 = v2.position

        # Find the nearest intersection for each position
        intersection1 = self._find_nearest_intersection(pos1)
        intersection2 = self._find_nearest_intersection(pos2)

        # If both are near the same intersection, they have line of sight
        if intersection1 is not None and intersection1 == intersection2:
            distance_to_intersection1 = self._distance_to_point(pos1, intersection1)
            distance_to_intersection2 = self._distance_to_point(pos2, intersection2)

            if distance_to_intersection1 < 50 and distance_to_intersection2 < 50:
                env_logger.debug(
                    f"Both positions near same intersection {intersection1}"
                )

                return True

        # Check if the line intersects any obstacles (buildings)
        for obs_x1, obs_y1, obs_x2, obs_y2 in self.obstacles:
            if self._line_intersects_box(
                x1, y1, x2, y2, obs_x1, obs_y1, obs_x2, obs_y2
            ):
                env_logger.debug(
                    f"Line of sight blocked by building: ({obs_x1}, {obs_y1}) to ({obs_x2}, {obs_y2})"
                )

                return False

        # Check if any vehicle blocks LOS
        if not wireless and all_vehicles:
            if self._vehicle_blocks_los(v1, v2, all_vehicles):
                env_logger.debug(f"Line of sight blocked by a vehicle")

                # Cache result for same-lane vehicles
                if same_lane:
                    self.los_cache[(v1.id, v2.id)] = (False, (pos1, pos2))

                return False

        # Cache result for same-lane vehicles
        if same_lane and not wireless:
            self.los_cache[(v1.id, v2.id)] = (True, (pos1, pos2))

        return True

    def _vehicle_blocks_los(self, v1, v2, all_vehicles):
        """Check if any vehicle blocks the line of sight between v1 and v2"""

        x1, y1 = v1.position
        x2, y2 = v2.position

        if self._check_on_same_road(v1, v2) and v1.lane_offset == v2.lane_offset:
            # If on same road, only check vehicles on that road between them
            for v_id, vehicle in all_vehicles.items():
                # Skip the vehicles being checked
                if v_id == v1.id or v_id == v2.id:
                    continue

                # Check if vehicle is on the same road
                if (
                    vehicle.current_road == v1.current_road
                    and vehicle.lane_offset == v1.lane_offset
                ):
                    v_x, v_y = vehicle.position

                    if v1.current_road["type"] == "horizontal":
                        # Check if vehicle is between v1 and v2 (x-coordinate)
                        min_x, max_x = min(x1, x2), max(x1, x2)
                        if min_x < v_x < max_x:
                            return True

                    else:  # vertical
                        # Check if vehicle is between v1 and v2 (y-coordinate)
                        min_y, max_y = min(y1, y2), max(y1, y2)
                        if min_y < v_y < max_y:
                            return True
        return False

    def clear_los_cache(self):
        """Clear the line of sight cache completely"""
        self.los_cache = {}

    def _find_nearest_intersection(self, position):
        """Find the nearest intersection to a given position"""
        min_distance = float("inf")
        nearest_intersection = None

        for intersection in self.intersections:
            distance = self._distance_to_point(position, intersection)
            if distance < min_distance:
                min_distance = distance
                nearest_intersection = intersection

        return nearest_intersection

    def _distance_to_point(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _check_on_same_road(self, v1: "Vehicle", v2: "Vehicle"):
        """Check if two positions are on the same road"""
        return v1.current_road == v2.current_road

    def _line_intersects_box(self, x1, y1, x2, y2, box_x1, box_y1, box_x2, box_y2):
        """
        Returns True if the line segment intersects the box, False otherwise.
        """
        # Define the line parameters
        dx = x2 - x1
        dy = y2 - y1

        # Initialize parameters
        p = [-dx, dx, -dy, dy]
        q = [x1 - box_x1, box_x2 - x1, y1 - box_y1, box_y2 - y1]
        u1, u2 = 0.0, 1.0

        # Process each edge of the box
        for i in range(4):
            if p[i] == 0:
                # Line is parallel to this edge
                if q[i] < 0:
                    # Line is outside the box
                    return False
            else:
                # Compute intersection parameter
                t = q[i] / p[i]
                if p[i] < 0:
                    # Line entering the box
                    u1 = max(u1, t)
                else:
                    # Line exiting the box
                    u2 = min(u2, t)

                if u1 > u2:
                    # Line doesn't intersect the box
                    return False

        # Check if intersection is within the line segment bounds
        if u1 <= u2 and u1 <= 1 and u2 >= 0:
            return True

        return False

    def get_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def plot(self):
        """Visualize the grid"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot roads
        for road in self.roads:
            start, end = road["start"], road["end"]
            ax.plot([start[0], end[0]], [start[1], end[1]], "k-", linewidth=2)

        # Plot obstacles
        for obs_x1, obs_y1, obs_x2, obs_y2 in self.obstacles:
            width = obs_x2 - obs_x1
            height = obs_y2 - obs_y1
            ax.add_patch(
                plt.Rectangle(
                    (obs_x1, obs_y1), width, height, color="maroon", alpha=0.5
                )
            )

        ax.set_xlim(0, self.total_size)
        ax.set_ylim(0, self.total_size)
        ax.set_aspect("equal")
        ax.set_title("Manhattan Grid with Obstacles")
        plt.show()
