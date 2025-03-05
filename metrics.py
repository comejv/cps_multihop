from logging import getLogger

import numpy as np

metrics_logger = getLogger("metrics")

RANGE_OF_INTEREST = 200  # meters


class MetricsCollector:
    """Collect simulation metrics"""

    def __init__(self):
        self.ear_values = []  # Environmental Awareness Ratio
        self.cbr_values = []  # Channel Busy Ratio
        self.aoi_values = []  # Age of Information
        self.cpm_sizes = []  # CPM message sizes

    def calculate_ear(self, vehicles, all_objects, current_time):
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

                if dist <= RANGE_OF_INTEREST:
                    objects_in_range += 1

                    # Check if object is perceived through sensors or V2X
                    is_perceived = False

                    # Through sensors
                    if obj_id in vehicle.objects_detected:
                        is_perceived = True

                    # Through V2X (in local environment model)
                    elif obj_id in vehicle.local_environment_model:
                        obj_data = vehicle.local_environment_model[obj_id]
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
        """Calculate Age of Information with simplified time handling"""
        aoi_values = []
        algorithm_specific_values = {}

        for vehicle in vehicles:
            if not vehicle.has_cps:
                continue

            # Get algorithm name for statistics
            algorithm_name = vehicle.algorithm.name
            if algorithm_name not in algorithm_specific_values:
                algorithm_specific_values[algorithm_name] = []

            # For each object in local environment model
            for obj_id, obj_data in vehicle.local_environment_model.items():
                # Skip objects from this vehicle (we care about received data)
                if obj_data["source_id"] == vehicle.id:
                    continue

                # Get when the object was last updated (from timestamp)
                update_time = obj_data["timestamp"]

                # Get when the vehicle received this information
                reception_time = vehicle.object_reception_times.get(obj_id)

                # Skip if we don't have reception time
                if reception_time is None:
                    continue

                # Ensure reception can't be before update (causality)
                reception_time = max(reception_time, update_time)

                # Calculate AOI in milliseconds
                aoi_ms = (reception_time - update_time) * 1000

                # Basic validation
                if 0 <= aoi_ms <= 10000:  # Skip extreme values
                    aoi_values.append(aoi_ms)
                    algorithm_specific_values[algorithm_name].append(aoi_ms)

        # Return mean AOI
        return np.mean(aoi_values) if aoi_values else 0.0

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
