from logging import getLogger

network_logger = getLogger("network")


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
        # Calculate transmission time based on message size
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

            # Get vehicles dict from simulation
            vehicles_dict = {v.id: v for v in self.vehicles}

            # Check: within range and has line of sight (including vehicle blocking)
            if (
                distance <= self.communication_range
                and self.environment.is_in_line_of_sight(
                    sender, vehicle, vehicles_dict, wireless=True
                )
            ):
                vehicle.receive_cpm(message, reception_time)

    def get_channel_busy_ratio(self):
        """Calculate Channel Busy Ratio and reset every window_size seconds"""
        if self.total_time > 0:
            self.current_cbr = self.channel_busy_time / self.total_time
        else:
            self.current_cbr = 0
        return self.current_cbr
