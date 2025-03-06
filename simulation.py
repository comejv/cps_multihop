import random
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from algenum import ForwardingAlgorithm
from environment import ManhattanGrid
from metrics import MetricsCollector
from network import WirelessNetwork
from utils import *
from vehicle import Vehicle

sim_logger = getLogger("simulation")


class SimulationStopException(Exception):
    """Exception raised to signal that the simulation should stop"""

    pass


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

    def run(self, simulation_time, dt=0.01, visualize=False):
        """Run simulation for a specified time with proper time sequencing"""
        num_steps = int(simulation_time / dt)

        # Reset simulation time
        self.current_time = 0.0

        # Reset network counters
        self.network.channel_busy_time = 0
        self.network.total_time = 0

        # Setup visualization if requested
        if visualize:
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            plt.ion()  # Turn on interactive mode

            # Add a button for stopping the simulation
            from matplotlib.widgets import Button

            stop_ax = plt.axes([0.81, 0.01, 0.1, 0.04])
            stop_button = Button(stop_ax, "Stop")

            # Define callback function
            def stop_simulation(event):
                nonlocal stop_requested
                stop_requested = True

                # Set the stop flag for all processes
                set_stop_flag()

                plt.close()  # Close the plot immediately

                # Raise an exception that will be caught in the main program
                raise SimulationStopException(
                    "Simulation stopped by user (Stop button)"
                )

            stop_button.on_clicked(stop_simulation)
            stop_requested = False

            # Draw static environment elements
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

            # Create legend
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

        vehicles_dict = {v.id: v for v in self.vehicles}

        try:
            for step in range(num_steps):
                # Check if stop has been requested
                if (
                    step % 10 == 0 and check_stop_flag()
                ):  # Check every 10 steps to reduce overhead
                    sim_logger.info("Stop flag detected, terminating simulation")
                    raise SimulationStopException("Stop requested via stop flag")

                # Update the current simulation time FIRST
                self.current_time = step * dt

                # Update total time for CBR calculation
                self.network.total_time += dt

                sim_logger.debug(f"Simulation step {step}, time = {self.current_time}")

                # 1. First, move all vehicles
                for vehicle in self.vehicles:
                    vehicle.move(dt)

                # 2. Then, have all vehicles sense objects
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
                                        vehicles_dict = {v.id: v for v in self.vehicles}
                                        if (
                                            dist <= vehicle.comm_range
                                            and self.environment.is_in_line_of_sight(
                                                vehicle,
                                                receiving_vehicle,
                                                vehicles_dict,
                                                wireless=True,
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
                        self.vehicles,
                        {v.id: v for v in self.vehicles},
                        self.current_time,
                    )
                    # Handle if ear_result is a tuple (ear, algorithm_ear)
                    if isinstance(ear_result, tuple):
                        ear, algorithm_ear = ear_result
                    else:
                        ear = ear_result

                    cbr = self.network.get_channel_busy_ratio()

                    aoi = self.metrics.calculate_aoi(self.vehicles, self.current_time)

                    self.metrics.record_cbr(cbr)

                    sim_logger.info(
                        f"Time: {self.current_time:.1f}s, EAR: {ear:.3f}, CBR: {cbr:.3f}, Avg AOI: {aoi:.3f}ms"
                    )

                    sim_logger.debug(
                        f"Cache stats : {self.environment.cache_hits}/{self.environment.cache_misses}"
                    )

                    # If we have algorithm-specific data, print that too
                    if isinstance(ear_result, tuple) and len(ear_result) > 1:
                        for alg, ear_val in algorithm_ear.items():
                            sim_logger.info(f"  - {alg} EAR: {ear_val:.3f}")

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
                            regular_vehicles_scatter.set_offsets(
                                np.column_stack([x, y])
                            )
                        else:
                            regular_vehicles_scatter.set_offsets(
                                np.column_stack([[], []])
                            )

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
                        plt.pause(0.6)  # Small pause to update plot

        except KeyboardInterrupt:
            sim_logger.critical("Simulation stopped by user (Keyboard Interrupt)")
            if visualize:
                plt.close()
            # Set the stop flag for all processes
            set_stop_flag()
            # Re-raise as SimulationStopException to propagate upward
            raise SimulationStopException("Simulation stopped by keyboard interrupt")

        except SimulationStopException as e:
            # Set the stop flag for all processes
            set_stop_flag()
            # Pass this up to be caught by the main program
            if visualize:
                plt.close()
            raise

        finally:
            if visualize:
                plt.close()

    def get_results(self):
        """Get simulation results"""
        return self.metrics.get_results()


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
    sim_logger.info("Simulation Results:", results)
