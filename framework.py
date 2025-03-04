import copy
import random
from enum import Enum

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEBUG_AOI = False  # Set to True for detailed AOI debugging
DEBUG_TIMESTAMPS = False  # Set to True for detailed timestamp debugging
DEBUG_LOS = False


class ForwardingAlgorithm(Enum):
    NO_FORWARDING = 1  # Baseline ETSI CPS
    GBC = 2  # Geographically-Scoped Broadcast
    MULTI_HOP = 3  # Proposed Application Layer Multi-Hop


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

        # Create buildings in each city block (not at intersections)
        for i in range(len(horizontal_roads) - 1):
            for j in range(len(vertical_roads) - 1):
                # Calculate block boundaries
                x1 = vertical_roads[j] + self.road_width / 2 + 20  # Add margin from road
                y1 = horizontal_roads[i] + self.road_width / 2 + 20
                x2 = vertical_roads[j + 1] - self.road_width / 2 - 20
                y2 = horizontal_roads[i + 1] - self.road_width / 2 - 20

                # Create one large building per block
                # This ensures communication across perpendicular roads is blocked
                obstacles.append((x1, y1, x2, y2))

        if DEBUG_LOS:
            print(f"Created {len(obstacles)} buildings in city blocks")

        return obstacles

    def is_in_line_of_sight(self, pos1, pos2):
        """Check if two positions have line of sight (not blocked by obstacles)"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Find the nearest intersection for each position
        intersection1 = self._find_nearest_intersection(pos1)
        intersection2 = self._find_nearest_intersection(pos2)

        # Check if both positions are on the same road
        on_same_road = self._check_on_same_road(pos1, pos2)

        # If they're on the same road, they have line of sight
        if on_same_road:
            if DEBUG_LOS:
                print(f"Positions on same road: {pos1} and {pos2}")
            return True

        # If both are near the same intersection, they have line of sight
        if intersection1 is not None and intersection1 == intersection2:
            distance_to_intersection1 = self._distance_to_point(pos1, intersection1)
            distance_to_intersection2 = self._distance_to_point(pos2, intersection2)

            # Define "near intersection" as within 75 meters
            if distance_to_intersection1 < 15 and distance_to_intersection2 < 15:
                if DEBUG_LOS:
                    print(f"Both positions near same intersection {intersection1}")
                return True

        # Otherwise check if the line intersects any obstacles
        for obs_x1, obs_y1, obs_x2, obs_y2 in self.obstacles:
            if self._line_intersects_box(
                x1, y1, x2, y2, obs_x1, obs_y1, obs_x2, obs_y2
            ):
                if DEBUG_LOS:
                    print(
                        f"Line of sight blocked by building: ({obs_x1}, {obs_y1}) to ({obs_x2}, {obs_y2})"
                    )
                return False

        return True

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

    def _check_on_same_road(self, pos1, pos2):
        """Check if two positions are on the same road"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Road width margin to determine if on road
        margin = self.road_width / 2 + 2  # Add a small buffer

        # Check if both are on the same horizontal road
        for road in [r for r in self.roads if r["type"] == "horizontal"]:
            road_y = road["start"][1]
            if abs(y1 - road_y) <= margin and abs(y2 - road_y) <= margin:
                return True

        # Check if both are on the same vertical road
        for road in [r for r in self.roads if r["type"] == "vertical"]:
            road_x = road["start"][0]
            if abs(x1 - road_x) <= margin and abs(x2 - road_x) <= margin:
                return True

        return False

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

        # Check if endpoints are inside the box - if so, it's not intersecting
        inside1 = box_x1 <= x1 <= box_x2 and box_y1 <= y1 <= box_y2
        inside2 = box_x1 <= x2 <= box_x2 and box_y1 <= y2 <= box_y2

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
                plt.Rectangle((obs_x1, obs_y1), width, height, color="maroon", alpha=0.5)
            )

        ax.set_xlim(0, self.total_size)
        ax.set_ylim(0, self.total_size)
        ax.set_aspect("equal")
        ax.set_title("Manhattan Grid with Obstacles")
        plt.show()


class Vehicle:
    """Vehicle class with movement and communication capabilities"""

    def __init__(
        self,
        vehicle_id,
        environment,
        has_cps=False,
        algorithm=ForwardingAlgorithm.NO_FORWARDING,
    ):
        self.id = vehicle_id
        self.environment = environment
        self.has_cps = has_cps
        self.algorithm = algorithm

        # Add these lines for road following
        self.current_road = random.choice(self.environment.roads)
        self.lane_offset = random.choice([-8, 8])  # Different lanes on the road

        # Position and movement
        self.position = self._init_position()
        self.speed = random.uniform(10, 20)  # m/s
        self.heading = random.choice([0, 90, 180, 270])  # Degrees
        self.lane = random.choice([0, 1])  # 0 = right lane, 1 = left lane

        # Sensing and communication
        self.sensing_range = 85  # meters
        self.comm_range = 300  # meters
        self.objects_detected = {}  # Objects detected by this vehicle

        # CPS related attributes
        self.local_environment_model = {}  # The LEM as specified in Algorithm 1
        self.cpm_buffer = []  # CPM buffer for storing received messages
        self.max_hop_count = 2  # Maximum hops for forwarding (as in the paper)
        self.last_update_time = {}  # Last time an object was included in a CPM

        # IMPORTANT: Initialize with empty dict - don't record any receptions before sim starts
        self.object_reception_times = {}

        # For periodic execution
        self.last_cpm_generation_time = 0
        self.cpm_generation_interval = 0.1

        # DCC parameters
        self.dcc_threshold = 0.8

        # NEW: Simulation start time tracking
        self.simulation_start_time = None  # Will be set on first update
        self.object_update_times = {}  # When objects were last updated (NEW FIELD)

    def _init_position(self):
        """Initialize vehicle at a random position on its assigned road"""
        if self.current_road["type"] == "horizontal":
            x = random.uniform(0, self.environment.total_size)
            y = self.current_road["start"][1] + self.lane_offset
            # Set heading based on lane offset
            self.heading = 0 if self.lane_offset > 0 else 180
            return (x, y)
        else:  # vertical
            x = self.current_road["start"][0] + self.lane_offset
            y = random.uniform(0, self.environment.total_size)
            # Set heading based on lane offset
            self.heading = 90 if self.lane_offset > 0 else 270
            return (x, y)

    def move(self, dt):
        """Move the vehicle along its assigned road"""
        x, y = self.position

        # Ensure vehicles have a valid heading for road movement
        # This makes sure no vehicles are immobile due to invalid heading
        if self.current_road["type"] == "horizontal":
            self.heading = 0 if self.lane_offset > 0 else 180
        else:  # vertical
            self.heading = 90 if self.lane_offset > 0 else 270

        # Move based on heading
        if self.heading == 0:  # East
            x += self.speed * dt
            if x > self.environment.total_size:
                x = 0  # Wrap around
        elif self.heading == 180:  # West
            x -= self.speed * dt
            if x < 0:
                x = self.environment.total_size  # Wrap around
        elif self.heading == 90:  # North
            y += self.speed * dt
            if y > self.environment.total_size:
                y = 0  # Wrap around
        elif self.heading == 270:  # South
            y -= self.speed * dt
            if y < 0:
                y = self.environment.total_size  # Wrap around

        # Always maintain the correct lane position
        if self.current_road["type"] == "horizontal":
            y = self.current_road["start"][1] + self.lane_offset
        else:  # vertical
            x = self.current_road["start"][0] + self.lane_offset

        self.position = (x, y)

    def sense_objects(self, all_objects, current_time):
        """Detect objects with proper timestamp handling for both new and updated objects"""

        if not self.has_cps:
            return {}

        detected = {}

        for obj_id, obj in all_objects.items():
            # Skip self
            if obj_id == self.id:
                continue

            dist = self.environment.get_distance(self.position, obj.position)

            # Check if object is within sensing range and has line of sight
            if dist <= self.sensing_range and self.environment.is_in_line_of_sight(
                self.position, obj.position
            ):
                # For new objects, set both timestamps to current time
                if obj_id not in self.objects_detected:
                    detected[obj_id] = {
                        "object_id": obj_id,
                        "position": obj.position,
                        "speed": obj.speed,
                        "heading": obj.heading,
                        "timestamp": current_time,
                        "source_id": self.id,
                        "hop_count": 0,
                        "original_detection_time": current_time,  # First detection
                        "update_time": current_time,  # Last update (NEW FIELD)
                    }

                    # Set first reception time for own detections
                    if obj_id not in self.object_reception_times:
                        self.object_reception_times[obj_id] = current_time

                    # Set first update time for own detections
                    if obj_id not in self.object_update_times:
                        self.object_update_times[obj_id] = current_time
                else:
                    # For existing objects, keep original detection time but update other fields
                    old_object = self.objects_detected[obj_id]
                    detected[obj_id] = {
                        "object_id": obj_id,
                        "position": obj.position,
                        "speed": obj.speed,
                        "heading": obj.heading,
                        "timestamp": current_time,  # Current time
                        "source_id": self.id,
                        "hop_count": 0,
                        "original_detection_time": old_object[
                            "original_detection_time"
                        ],  # Keep original
                        "update_time": current_time,  # New update time (NEW FIELD)
                    }

                    # Always update the update time
                    self.object_update_times[obj_id] = current_time

        self.objects_detected = detected
        return detected

    def receive_cpm(self, cpm, reception_time):
        """Process received CPM with proper update time tracking"""
        if not self.has_cps:
            return

        # Process objects from the CPM
        for obj_data in cpm["objects"]:
            obj_id = obj_data["object_id"]

            # Create a deep copy to avoid modifying the original
            obj_data_copy = copy.deepcopy(obj_data)

            # Ensure original_detection_time is preserved
            if "original_detection_time" not in obj_data_copy:
                if DEBUG_TIMESTAMPS:
                    print(
                        f"WARNING: Missing original_detection_time for object {obj_id}"
                    )
                obj_data_copy["original_detection_time"] = obj_data_copy["timestamp"]

            # Ensure update_time is set - use timestamp if not explicitly set
            if "update_time" not in obj_data_copy:
                if DEBUG_TIMESTAMPS:
                    print(
                        f"DEBUG: Setting update_time for object {obj_id} to its timestamp"
                    )
                obj_data_copy["update_time"] = obj_data_copy["timestamp"]

            # CRITICAL: Ensure reception time is never earlier than detection time
            valid_reception_time = max(
                reception_time, obj_data_copy["original_detection_time"]
            )

            # Record FIRST reception time if not already recorded
            if obj_id not in self.object_reception_times:
                self.object_reception_times[obj_id] = valid_reception_time
                if DEBUG_TIMESTAMPS:
                    print(
                        f"DEBUG: First reception of object {obj_id} at time {valid_reception_time}"
                    )

            # Always update the last update time
            self.object_update_times[obj_id] = valid_reception_time
            if DEBUG_TIMESTAMPS:
                print(f"DEBUG: Updated object {obj_id} at time {valid_reception_time}")

            # Update local environment model if newer information is available
            if obj_id not in self.local_environment_model or (
                self.local_environment_model[obj_id]["update_time"]
                < obj_data_copy["update_time"]
                and obj_data_copy["hop_count"] < self.max_hop_count
            ):
                # Update reception and valid times
                obj_data_copy["reception_time"] = valid_reception_time
                self.local_environment_model[obj_id] = obj_data_copy

    def run_cps_algorithm(self, current_time, network):
        """Run the CPS algorithm with proper update time tracking"""
        if not self.has_cps:
            return None

        # Check if it's time to run the algorithm
        time_since_last_execution = current_time - self.last_cpm_generation_time
        if time_since_last_execution < self.cpm_generation_interval:
            return None

        # Update the last execution time
        self.last_cpm_generation_time = current_time

        # Update local environment model with own detected objects
        for obj_id, obj_data in self.objects_detected.items():
            # Make sure timestamps are set for own objects
            obj_data["timestamp"] = current_time
            obj_data["update_time"] = current_time  # Always update with latest info

            # Preserve original detection time
            if "original_detection_time" not in obj_data:
                obj_data["original_detection_time"] = obj_data.get(
                    "timestamp", current_time
                )

            self.local_environment_model[obj_id] = obj_data

            # Ensure reception and update times are set for own objects
            if obj_id not in self.object_reception_times:
                self.object_reception_times[obj_id] = current_time

            # Always update this field for own objects
            self.object_update_times[obj_id] = current_time

        # Create new CPM
        new_cpm = {
            "sender_id": self.id,
            "timestamp": current_time,
            "position": self.position,
            "objects": [],
        }

        # Add objects to CPM based on kinematic change
        for obj_id, obj_data in self.local_environment_model.items():
            # Skip if hop count is already at max
            if obj_data["hop_count"] >= self.max_hop_count:
                continue

            # Check if we should include this object based on kinematic update rules
            should_include = False

            # Kinematic change trigger logic
            if obj_id not in self.last_update_time:
                should_include = True
            else:
                last_time = self.last_update_time[obj_id]["time"]
                last_pos = self.last_update_time[obj_id]["position"]
                last_speed = self.last_update_time[obj_id]["speed"]
                last_heading = self.last_update_time[obj_id]["heading"]

                # ETSI kinematic update rules
                time_diff = current_time - last_time
                pos_diff = self.environment.get_distance(last_pos, obj_data["position"])
                speed_diff = abs(last_speed - obj_data["speed"])
                heading_diff = abs(last_heading - obj_data["heading"])

                if (
                    time_diff > 1.0  # More than 1 second
                    or pos_diff > 4.0  # Position change > 4m
                    or speed_diff > 4.0  # Speed change > 4 m/s
                    or heading_diff > 4.0
                ):  # Heading change > 4°
                    should_include = True

            if should_include:
                # Update last inclusion time
                self.last_update_time[obj_id] = {
                    "time": current_time,
                    "position": obj_data["position"],
                    "speed": obj_data["speed"],
                    "heading": obj_data["heading"],
                }

                # Create a copy with updated hop count for forwarding
                obj_data_copy = copy.deepcopy(obj_data)

                # CRITICAL: Preserve original_detection_time during forwarding
                # Make sure original_detection_time exists and is never lost
                if "original_detection_time" not in obj_data_copy:
                    if DEBUG_TIMESTAMPS:
                        print(
                            f"WARNING: Missing original_detection_time for object {obj_id} during forwarding"
                        )
                    obj_data_copy["original_detection_time"] = obj_data["timestamp"]

                # Important: Only increment hop count for objects from other vehicles
                if obj_data["source_id"] != self.id:
                    obj_data_copy["hop_count"] += 1

                    # Update timestamp to current time for forwarding
                    obj_data_copy["timestamp"] = current_time

                # Add object to CPM
                new_cpm["objects"].append(obj_data_copy)

        # Check DCC condition before sending
        cbr = network.get_channel_busy_ratio()
        if cbr < self.dcc_threshold:
            return new_cpm if new_cpm["objects"] else None

        return None


class WirelessNetwork:
    """Simplified wireless network simulation with basic range and obstacle checks"""

    def __init__(self, environment, vehicles):
        self.environment = environment
        self.vehicles = vehicles
        self.channel_busy_time = 0
        self.total_time = 0
        self.communication_range = 300  # meters
        self.cbr_window_size = 5  # 5-second window for CBR calculation
        self.last_cbr_reset_time = 0  # Last time CBR counters were reset
        self.current_cbr = 0  # Store the current CBR value

    def simulate_transmission(self, sender, message, current_time, dt):
        """Simulate message transmission with simplified model - only range and LOS checks"""
        # Calculate transmission time based on message size (keep this for CBR calculation)
        message_size_bytes = 50 + len(message["objects"]) * 10
        bit_rate = 6e6  # 6 Mbit/s
        transmission_time = (message_size_bytes * 8) / bit_rate

        # Update channel busy time
        self.channel_busy_time += transmission_time

        # Ensure reception_time is always at least current_time
        reception_time = max(current_time, current_time + transmission_time)

        # For each vehicle, check if it can receive the message
        for vehicle in self.vehicles:
            if hasattr(sender, "id") and vehicle.id == sender.id:
                continue  # Skip sender

            # Calculate distance
            distance = self.environment.get_distance(sender.position, vehicle.position)

            # Simple check: within range and has line of sight
            if (
                distance <= self.communication_range
                and self.environment.is_in_line_of_sight(
                    sender.position, vehicle.position
                )
            ):
                # Vehicle receives the message
                vehicle.receive_cpm(message, reception_time)

    def get_channel_busy_ratio(self):
        """Calculate Channel Busy Ratio and reset every window_size seconds"""
        if self.total_time > 0:
            self.current_cbr = self.channel_busy_time / self.total_time
        else:
            self.current_cbr = 0
        return self.current_cbr


class MetricsCollector:
    """Collect simulation metrics"""

    def __init__(self):
        self.ear_values = []  # Environmental Awareness Ratio
        self.cbr_values = []  # Channel Busy Ratio
        self.aoi_values = []  # Age of Information
        self.cpm_sizes = []  # CPM message sizes

    def calculate_ear(self, vehicles, all_objects, current_time, aoi_threshold=1.0):
        """Calculate Environmental Awareness Ratio with improved performance"""
        total_objects_in_range = 0
        perceived_objects = 0

        # Pre-compute distances for performance (limit N²)
        distance_cache = {}

        # First phase: Calculate distances between all vehicles once
        for vehicle in vehicles:
            if not vehicle.has_cps:
                continue

            vehicle_id = vehicle.id
            vehicle_pos = vehicle.position

            for obj_id, obj in all_objects.items():
                if obj_id == vehicle_id:
                    continue

                # Cache the distance calculation
                cache_key = (vehicle_id, obj_id)
                distance_cache[cache_key] = vehicle.environment.get_distance(
                    vehicle_pos, obj.position
                )

        # Second phase: Use cached distances for EAR calculation
        for vehicle in vehicles:
            if not vehicle.has_cps:
                continue

            vehicle_id = vehicle.id
            objects_in_range = 0
            objects_perceived = 0

            for obj_id, obj in all_objects.items():
                if obj_id == vehicle_id:
                    continue

                # Use cached distance
                dist = distance_cache.get((vehicle_id, obj_id), float("inf"))

                if dist <= vehicle.sensing_range:
                    objects_in_range += 1

                    # Check if object is perceived through sensors or V2X
                    is_perceived = False

                    # Through sensors
                    if obj_id in vehicle.objects_detected:
                        is_perceived = True

                    # Through V2X (in local environment model)
                    elif obj_id in vehicle.local_environment_model:
                        obj_data = vehicle.local_environment_model[obj_id]
                        if current_time - obj_data["timestamp"] <= aoi_threshold:
                            is_perceived = True

                    if is_perceived:
                        objects_perceived += 1

            # Update overall counts
            total_objects_in_range += objects_in_range
            perceived_objects += objects_perceived

        # Calculate overall EAR (with safe division)
        ear = perceived_objects / max(1, total_objects_in_range)

        # Store the EAR value
        self.ear_values.append(ear)

        return ear

    def record_cbr(self, cbr):
        """Record Channel Busy Ratio"""
        self.cbr_values.append(cbr)

    def calculate_aoi(self, vehicles, current_time):
        """Calculate Age of Information with optimized performance"""
        # Create simplified tracking for AOI values
        aoi_values = []
        algorithm_specific_values = {}

        # Count valid AOI values for reporting
        valid_count = 0
        invalid_count = 0

        for vehicle in vehicles:
            if not vehicle.has_cps:
                continue

            # Get algorithm name for final statistics
            algorithm_name = vehicle.algorithm.name
            if algorithm_name not in algorithm_specific_values:
                algorithm_specific_values[algorithm_name] = []

            # For each object in local environment model
            for obj_id, obj_data in vehicle.local_environment_model.items():
                # Skip objects from this vehicle (we care about received data)
                if obj_data["source_id"] == vehicle.id:
                    continue

                # KEY OPTIMIZATION: Simplified access to update and reception times
                # Get update time - when the object state was last updated
                update_time = obj_data.get("update_time", obj_data.get("timestamp", 0))

                # Get reception time - when this vehicle received the info
                reception_time = vehicle.object_update_times.get(
                    obj_id, vehicle.object_reception_times.get(obj_id, None)
                )

                # Skip if we don't have reception time
                if reception_time is None:
                    continue

                # Ensure reception can't be before update (causality)
                reception_time = max(reception_time, update_time)

                # Calculate AOI in milliseconds
                aoi_ms = (reception_time - update_time) * 1000

                # Basic validation
                if aoi_ms < 0 or aoi_ms > 10000:  # Skip invalid values
                    invalid_count += 1
                    continue

                # Store valid AOI
                aoi_values.append(aoi_ms)
                algorithm_specific_values[algorithm_name].append(aoi_ms)
                valid_count += 1

        # For final statistics, we only need basic results during simulation
        if valid_count == 0:
            if DEBUG_AOI:
                print("DEBUG AOI: No valid AOI values found")
            return 0.0

        # OPTIMIZATION: Only compute the mean for real-time feedback
        mean_aoi = np.mean(aoi_values)

        # Store all valid values for detailed analysis later
        self.aoi_values.extend(aoi_values)

        # Simple logging only for overall stats
        if DEBUG_AOI and valid_count > 0:
            print(f"DEBUG AOI: {valid_count} valid values, {invalid_count} invalid")
            print(f"DEBUG AOI: Mean AOI = {mean_aoi:.1f}ms")

        return mean_aoi

    def record_cpm_size(self, cpm):
        """Record CPM message size (number of objects)"""
        if cpm and "objects" in cpm:
            self.cpm_sizes.append(len(cpm["objects"]))

    def get_results(self):
        """Get summary of results"""
        results = {
            "EAR": {
                "mean": np.mean(self.ear_values) if self.ear_values else 0,
                "median": np.median(self.ear_values) if self.ear_values else 0,
            },
            "CBR": {
                "mean": np.mean(self.cbr_values) if self.cbr_values else 0,
                "median": np.median(self.cbr_values) if self.cbr_values else 0,
            },
            "AOI": {
                "mean": np.mean(self.aoi_values) if self.aoi_values else 0,
                "median": np.median(self.aoi_values) if self.aoi_values else 0,
            },
            "CPM_Size": {
                "mean": np.mean(self.cpm_sizes) if self.cpm_sizes else 0,
                "median": np.median(self.cpm_sizes) if self.cpm_sizes else 0,
            },
        }
        return results

    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # EAR plot
        axes[0, 0].plot(self.ear_values)
        axes[0, 0].set_title("Environmental Awareness Ratio")
        axes[0, 0].set_xlabel("Simulation time step")
        axes[0, 0].set_ylabel("EAR")
        axes[0, 0].set_ylim(0, 1)

        # CBR plot
        axes[0, 1].plot(self.cbr_values)
        axes[0, 1].set_title("Channel Busy Ratio")
        axes[0, 1].set_xlabel("Simulation time step")
        axes[0, 1].set_ylabel("CBR")
        axes[0, 1].set_ylim(0, 1)

        # AOI histogram
        if self.aoi_values:
            axes[1, 0].hist(self.aoi_values, bins=20)
            axes[1, 0].set_title("Age of Information Distribution")
            axes[1, 0].set_xlabel("AOI (s)")
            axes[1, 0].set_ylabel("Frequency")

        # CPM size histogram
        if self.cpm_sizes:
            axes[1, 1].hist(self.cpm_sizes, bins=10)
            axes[1, 1].set_title("CPM Size Distribution")
            axes[1, 1].set_xlabel("Number of objects per CPM")
            axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()


class Simulation:
    """Main simulation class"""

    def __init__(self, config):
        self.config = config
        # Update to pass road_width
        self.environment = ManhattanGrid(grid_size=3, block_length=250, road_width=16)
        self.vehicles = []
        self.current_time = 0
        self.metrics = MetricsCollector()

        # Create vehicles
        self._create_vehicles()

        # Create network
        self.network = WirelessNetwork(self.environment, self.vehicles)

    def _create_vehicles(self):
        """Create vehicles based on density and penetration rate"""
        # Calculate number of vehicles based on density
        # Use the values from the paper: 30 vehicles/km (low) and 60 vehicles/km (high)
        total_road_length = 0
        for road in self.environment.roads:
            start, end = road["start"], road["end"]
            length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            total_road_length += length

        total_road_length_km = total_road_length / 1000
        num_vehicles = int(self.config["vehicle_density"] * total_road_length_km)

        # Create vehicles
        for i in range(num_vehicles):
            has_cps = random.random() < self.config["penetration_rate"]
            algorithm = self.config["algorithm"] if has_cps else None

            vehicle = Vehicle(i, self.environment, has_cps, algorithm)
            self.vehicles.append(vehicle)

    # Update the visualization part of the run method
    def run(self, simulation_time, dt=0.01, visualize=False):
        """Run simulation for a specified time with proper time sequencing"""
        num_steps = int(simulation_time / dt)

        # Reset simulation time
        self.current_time = 0.0

        # Reset network counters
        self.network.channel_busy_time = 0
        self.network.total_time = 0

        # Initialize simulation start time for all vehicles
        for vehicle in self.vehicles:
            vehicle.simulation_start_time = 0.0
            # Reset all vehicle tracking dictionaries to ensure clean start
            vehicle.object_reception_times = {}
            vehicle.object_update_times = {}
            vehicle.local_environment_model = {}
            vehicle.objects_detected = {}
            vehicle.last_update_time = {}
            vehicle.last_cpm_generation_time = 0

        # Setup visualization if requested
        if visualize:
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            plt.ion()  # Turn on interactive mode

            # Draw static environment elements - UPDATED: Draw roads with actual width
            for road in self.environment.roads:
                start, end = road["start"], road["end"]
                # Calculate road rectangle coordinates
                half_width = self.environment.road_width / 2

                if road["type"] == "horizontal":
                    # Horizontal road: width is in y-direction
                    x = [start[0], end[0]]
                    y_lower = [start[1] - half_width, end[1] - half_width]
                    y_upper = [start[1] + half_width, end[1] + half_width]

                    # Draw road as a filled polygon (rectangle)
                    ax.fill(
                        [x[0], x[1], x[1], x[0]],
                        [y_lower[0], y_lower[1], y_upper[1], y_upper[0]],
                        color="gray",
                        alpha=0.7,
                    )

                    # Draw center line
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        "white",
                        linestyle="--",
                        linewidth=1,
                    )
                else:  # vertical
                    # Vertical road: width is in x-direction
                    y = [start[1], end[1]]
                    x_lower = [start[0] - half_width, end[0] - half_width]
                    x_upper = [start[0] + half_width, end[0] + half_width]

                    # Draw road as a filled polygon (rectangle)
                    ax.fill(
                        [x_lower[0], x_lower[1], x_upper[1], x_upper[0]],
                        [y[0], y[1], y[1], y[0]],
                        color="gray",
                        alpha=0.7,
                    )

                    # Draw center line
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        "white",
                        linestyle="--",
                        linewidth=1,
                    )

            # Draw obstacles
            for obs_x1, obs_y1, obs_x2, obs_y2 in self.environment.obstacles:
                width = obs_x2 - obs_x1
                height = obs_y2 - obs_y1

                # Draw building with a darker color and border
                ax.add_patch(
                    plt.Rectangle(
                        (obs_x1, obs_y1),
                        width,
                        height,
                        facecolor="darkred",
                        edgecolor="black",
                        alpha=0.8,
                        linewidth=1,
                    )
                )

            # Set plot limits
            ax.set_xlim(0, self.environment.total_size)
            ax.set_ylim(0, self.environment.total_size)
            ax.set_aspect("equal")
            ax.set_title("VANET Simulation with Realistic Urban Environment")

            # Create legend with updated descriptions
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="blue",
                    markersize=10,
                    label="Regular Vehicle",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    label="CPS Vehicle",
                ),
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="darkred",
                    alpha=0.8,
                    label="Buildings",
                ),
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="gray",
                    alpha=0.7,
                    label="Roads",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

            # Initialize scatter plots
            regular_vehicles_scatter = ax.scatter([], [], c="blue", s=30)
            cps_vehicles_scatter = ax.scatter([], [], c="green", s=50)

            # For communication visualization
            comm_lines = []

            # Add a text box for simulation stats
            stats_text = ax.text(
                0.02,
                0.98,
                "",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        for step in range(num_steps):
            # Update the current simulation time FIRST
            self.current_time = step * dt

            # Update total time for CBR calculation
            self.network.total_time += dt

            if DEBUG_TIMESTAMPS:
                print(f"\nSimulation step {step}, time = {self.current_time}")

            # 1. First, move all vehicles
            for vehicle in self.vehicles:
                vehicle.move(dt)

            # 2. Then, have all vehicles sense objects
            vehicles_dict = {v.id: v for v in self.vehicles}
            for vehicle in self.vehicles:
                # Pass the current time to ensure consistent timestamps
                vehicle.sense_objects(vehicles_dict, self.current_time)

            # 3. Finally, process communications - this must come after sensing
            recent_communications = []  # Track for visualization

            for vehicle in self.vehicles:
                if vehicle.has_cps:
                    # Run the CPS algorithm
                    cpm = vehicle.run_cps_algorithm(self.current_time, self.network)

                    if cpm:
                        # Record CPM size
                        self.metrics.record_cpm_size(cpm)

                        # Store communication for visualization
                        if visualize:
                            for receiving_vehicle in self.vehicles:
                                if (
                                    receiving_vehicle.id != vehicle.id
                                    and receiving_vehicle.has_cps
                                ):
                                    dist = self.environment.get_distance(
                                        vehicle.position, receiving_vehicle.position
                                    )
                                    if (
                                        dist <= vehicle.comm_range
                                        and self.environment.is_in_line_of_sight(
                                            vehicle.position, receiving_vehicle.position
                                        )
                                    ):
                                        recent_communications.append(
                                            (
                                                vehicle.position,
                                                receiving_vehicle.position,
                                            )
                                        )

                        # Simulate transmission with current_time
                        self.network.simulate_transmission(
                            vehicle, cpm, self.current_time, dt
                        )

            # Collect metrics every 1 second
            if step % int(1 / dt) == 0:
                ear_result = self.metrics.calculate_ear(
                    self.vehicles, {v.id: v for v in self.vehicles}, self.current_time
                )
                # Handle if ear_result is a tuple (ear, algorithm_ear)
                if isinstance(ear_result, tuple):
                    ear, algorithm_ear = ear_result
                else:
                    ear = ear_result

                cbr = self.network.get_channel_busy_ratio()

                # Add this debug line to track data collection
                if DEBUG_AOI:
                    print(
                        f"\nDEBUG: Collecting AOI metrics at time {self.current_time}s"
                    )

                aoi = self.metrics.calculate_aoi(self.vehicles, self.current_time)

                self.metrics.record_cbr(cbr)

                print(
                    f"Time: {self.current_time:.1f}s, EAR: {ear:.3f}, CBR: {cbr:.3f}, Avg AOI: {aoi:.3f}ms"
                )

                # If we have algorithm-specific data, print that too
                if isinstance(ear_result, tuple) and len(ear_result) > 1:
                    for alg, ear_val in algorithm_ear.items():
                        if ear_val < 1.0:  # Only print if there are actually objects
                            print(f"  - {alg} EAR: {ear_val:.3f}")

                # Update visualization every 1 second if enabled
                if visualize:
                    # Clear previous communication lines
                    for line in comm_lines:
                        try:
                            line.remove()
                        except:
                            pass  # In case the line was already removed
                    comm_lines = []

                    # Update vehicle positions
                    regular_vehicles = [
                        (v.position[0], v.position[1])
                        for v in self.vehicles
                        if not v.has_cps
                    ]
                    cps_vehicles = [
                        (v.position[0], v.position[1])
                        for v in self.vehicles
                        if v.has_cps
                    ]

                    if regular_vehicles:
                        x, y = zip(*regular_vehicles)
                        regular_vehicles_scatter.set_offsets(np.column_stack([x, y]))
                    else:
                        regular_vehicles_scatter.set_offsets(np.column_stack([[], []]))

                    if cps_vehicles:
                        x, y = zip(*cps_vehicles)
                        cps_vehicles_scatter.set_offsets(np.column_stack([x, y]))
                    else:
                        cps_vehicles_scatter.set_offsets(np.column_stack([[], []]))

                    # Visualize recent communications (limit to last 20 to avoid clutter)
                    for i, (sender_pos, receiver_pos) in enumerate(
                        recent_communications
                    ):
                        line = ax.plot(
                            [sender_pos[0], receiver_pos[0]],
                            [sender_pos[1], receiver_pos[1]],
                            "r-",
                            alpha=0.3,
                            linewidth=1,
                        )[0]
                        comm_lines.append(line)

                    # Update stats text
                    stats_text.set_text(
                        f"Time: {self.current_time:.1f}s\nEAR: {ear:.3f}\nCBR: {cbr:.3f}\nAOI: {aoi:.3f}ms"
                    )

                    plt.draw()
                    plt.pause(0.01)  # Small pause to update plot
        plt.close()

    def get_results(self):
        """Get simulation results"""
        return self.metrics.get_results()

    def plot_results(self):
        """Plot simulation results"""
        self.metrics.plot_results()


# Example usage:
if __name__ == "__main__":
    # Configuration for replicating the paper's simulation
    config = {
        "vehicle_density": 30,  # Low density (30 vehicles/km)
        "penetration_rate": 0.1,  # 10% penetration rate
        "algorithm": ForwardingAlgorithm.MULTI_HOP,  # Proposed algorithm
    }

    # Create and run simulation
    sim = Simulation(config)
    sim.run(simulation_time=15)  # 15 seconds per run

    # Get and plot results
    results = sim.get_results()
    print("Simulation Results:", results)
    sim.plot_results()
