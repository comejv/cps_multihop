import copy
import random
from logging import getLogger

from algenum import ForwardingAlgorithm

vehicle_logger = getLogger("vehicle")


class Vehicle:
    """Vehicle class with implementation closely matching Wolff algorithm"""

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

        # Position and movement
        self.current_road = random.choice(self.environment.roads)
        self.lane_offset = random.choice([-8, 8])
        self.position = self._init_position()
        self.speed = random.uniform(10, 20)  # m/s
        self.heading = random.choice([0, 90, 180, 270])  # Degrees

        # Sensing and communication
        self.sensing_range = 85  # meters
        self.comm_range = 200  # meters
        self.objects_detected = {}  # Objects detected by this vehicle

        # === ALGORITHM 1 INITIALIZATION PART ===
        # Initialize Local Environment Model (LEM): LEM ← ∅
        self.local_environment_model = {}  # The LEM from the paper

        # Initialize CPM Buffer: CPM_Buffer ← ∅
        self.cpm_buffer = []  # Added explicit CPM buffer from the paper

        # Define maximum hop count: MAX_HOP_COUNT
        self.max_hop_count = 2  # Maximum hops for forwarding as in paper
        # =====================================

        # Time related fields
        self.object_reception_times = {}  # When info about each object was received
        self.kinematic_update_tracker = {}  # For kinematic trigger checking

        # CPS parameters
        self.aoi_threshold = 1  # second
        self.last_cpm_generation_time = 0
        self.cpm_generation_interval = 0.1
        self.dcc_threshold = 0.7

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
        """Detect objects with simplified time handling"""
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
                self, obj, all_objects
            ):
                # Create object data with simplified time fields
                detected[obj_id] = {
                    "object_id": obj_id,
                    "position": obj.position,
                    "speed": obj.speed,
                    "heading": obj.heading,
                    "timestamp": current_time,
                    "source_id": self.id,
                    "hop_count": 0,
                }

                # Set reception time for own detections
                if obj_id not in self.object_reception_times:
                    self.object_reception_times[obj_id] = current_time

        self.objects_detected = detected
        return detected

    def receive_cpm(self, cpm, reception_time):
        """
        Process received CPM by adding to buffer
        Maps to 'CPM_Buffer' in Algorithm 1
        """
        if not self.has_cps:
            return

        # Instead of processing immediately, add to buffer as in Algorithm 1
        self.cpm_buffer.append((cpm, reception_time))

    def _process_cpm_buffer(self, current_time):
        """
        Process all CPMs in the buffer
        Maps to the 'While not CPM_Buffer.isEmpty()' loop in Algorithm 1
        """
        while self.cpm_buffer:
            cpm, reception_time = self.cpm_buffer.pop(0)

            # Process objects from the CPM
            for obj_data in cpm["objects"]:
                obj_id = obj_data["object_id"]

                # Apply forwarding algorithm filters
                # For NO_FORWARDING, we only accept objects with hop_count=0 (directly sensed)
                if (
                    self.algorithm == ForwardingAlgorithm.NO_FORWARDING
                    and obj_data["hop_count"] > 0
                ):
                    continue

                # For GBC, network layer should handle forwarding
                if (
                    self.algorithm == ForwardingAlgorithm.GBC
                    and obj_data["hop_count"] > 0
                ):
                    continue

                # Record reception time for AOI calculation
                self.object_reception_times[obj_id] = reception_time

                # Update LEM if object is not in LEM OR if LEM[objID].timestamp < object.timestamp
                # AND object.hopCount < MAX_HOP_COUNT (from Algorithm 1)
                if obj_id not in self.local_environment_model or (
                    self.local_environment_model[obj_id]["timestamp"]
                    < obj_data["timestamp"]
                    and obj_data["hop_count"] < self.max_hop_count
                ):
                    self.local_environment_model[obj_id] = obj_data

    def _kinematic_change_trigger(self, obj_id, obj_data, current_time):
        """
        Check if object meets kinematic criteria for inclusion in CPM
        Maps to object.kinematicChangeTrigger() in Algorithm 1
        """
        if obj_id not in self.kinematic_update_tracker:
            # First time seeing this object
            self.kinematic_update_tracker[obj_id] = {
                "time": current_time,
                "position": obj_data["position"],
                "speed": obj_data["speed"],
                "heading": obj_data["heading"],
            }
            return True

        last_update = self.kinematic_update_tracker[obj_id]

        # ETSI kinematic update rules
        time_diff = current_time - last_update["time"]
        pos_diff = self.environment.get_distance(
            last_update["position"], obj_data["position"]
        )
        speed_diff = abs(last_update["speed"] - obj_data["speed"])
        heading_diff = abs(last_update["heading"] - obj_data["heading"])

        if (
            time_diff > 1.0  # More than 1 second
            or pos_diff > 4.0  # Position change > 4m
            or speed_diff > 4.0  # Speed change > 4 m/s
            or heading_diff > 4.0  # Heading change > 4°
        ):
            # Update last inclusion time
            self.kinematic_update_tracker[obj_id] = {
                "time": current_time,
                "position": obj_data["position"],
                "speed": obj_data["speed"],
                "heading": obj_data["heading"],
            }
            return True

        return False

    def _dcc_send_condition(self, network):
        """
        Check if DCC allows sending
        Maps to DCC_SEND_CONDITION() in Algorithm 1
        """
        cbr = network.get_channel_busy_ratio()
        return cbr < self.dcc_threshold

    def run_cps_algorithm(self, current_time, network):
        """
        Run the CPS algorithm following Algorithm 1 structure
        """
        if not self.has_cps:
            return None

        # === ALGORITHM 1 MAIN LOOP STRUCTURE ===

        # Check if it's time to run the algorithm (PERIODIC_EXECUTION_WAIT())
        time_since_last_execution = current_time - self.last_cpm_generation_time
        if time_since_last_execution < self.cpm_generation_interval:
            return None

        # Update the last execution time
        self.last_cpm_generation_time = current_time

        # Process CPM Buffer (first part of the While loop in Algorithm 1)
        self._process_cpm_buffer(current_time)

        # Remove stale objects from LEM (not in Algorithm 1 but good practice)
        stale_object_ids = []
        for obj_id, obj_data in self.local_environment_model.items():
            time_since_update = current_time - obj_data["timestamp"]
            if time_since_update > self.aoi_threshold:
                stale_object_ids.append(obj_id)

        for obj_id in stale_object_ids:
            del self.local_environment_model[obj_id]
            if obj_id in self.object_reception_times:
                del self.object_reception_times[obj_id]

        # Update LEM with own detected objects (implicit in Algorithm 1)
        for obj_id, obj_data in self.objects_detected.items():
            obj_data["timestamp"] = current_time
            self.local_environment_model[obj_id] = obj_data
            self.object_reception_times[obj_id] = current_time

        # Create new CPM (CREATE_NEW_CPM())
        new_cpm = {
            "sender_id": self.id,
            "timestamp": current_time,
            "position": self.position,
            "objects": [],
        }

        # Add objects meeting criteria (For loop in Algorithm 1)
        for obj_id, obj_data in self.local_environment_model.items():
            # Filter based on algorithm
            if (
                self.algorithm == ForwardingAlgorithm.NO_FORWARDING
                or self.algorithm == ForwardingAlgorithm.GBC
            ) and obj_data["source_id"] != self.id:
                continue

            # Check max hop count (directly from Algorithm 1)
            if obj_data["hop_count"] >= self.max_hop_count:
                continue

            # Check kinematic trigger (object.kinematicChangeTrigger())
            if self._kinematic_change_trigger(obj_id, obj_data, current_time):
                # Create a copy with updated hop count for forwarding
                obj_data_copy = obj_data.copy()

                # Increment hop count for MULTI_HOP mode
                if (
                    self.algorithm == ForwardingAlgorithm.MULTI_HOP
                    and obj_data["source_id"] != self.id
                ):
                    obj_data_copy["hop_count"] += 1
                    obj_data_copy["timestamp"] = current_time

                # Add to CPM
                new_cpm["objects"].append(obj_data_copy)

        # Check DCC condition before sending (DCC_SEND_CONDITION())
        if self._dcc_send_condition(network):
            return new_cpm if new_cpm["objects"] else None

        return None
