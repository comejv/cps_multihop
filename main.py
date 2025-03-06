import argparse
import concurrent.futures
import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from log_config import set_component_level, setup_logging
from simulation import ForwardingAlgorithm, Simulation

logger = logging.getLogger("main")


def run_single_simulation(config, simulation_time, visualize):
    logger.info(
        f"Running simulations for density={config['vehicle_density']}, rate={config['penetration_rate']}, algorithm={config['algorithm']}"
    )

    sim = Simulation(config)
    sim.run(simulation_time=simulation_time, visualize=visualize, dt=0.1)
    return sim.get_results()


def run_experiment_parallel(
    vehicle_densities,
    penetration_rates,
    algorithms,
    num_runs=10,
    simulation_time=15,
    visualize=False,
    max_workers=None,
    mute=True,
):
    """Parallelized version of run_experiment using ProcessPoolExecutor"""
    results = []
    configs = []

    # Create all configurations first
    for density in vehicle_densities:
        for rate in penetration_rates:
            for algorithm in algorithms:
                for run in range(num_runs):
                    config = {
                        "vehicle_density": density,
                        "penetration_rate": rate,
                        "algorithm": algorithm,
                    }
                    configs.append((config, algorithm.name, density, rate, run))

    logger.info(f"Starting experiment with {len(configs)} configurations")
    logger.info(f"Vehicle densities: {vehicle_densities}")
    logger.info(f"Penetration rates: {penetration_rates}")
    logger.info(f"Algorithms: {[alg.name for alg in algorithms]}")

    # Run simulations in parallel
    sim_func = partial(
        run_single_simulation, simulation_time=simulation_time, visualize=False
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(sim_func, config) for config, _, _, _, _ in configs]

        for future, (config, algorithm_name, density, rate, run) in zip(
            concurrent.futures.as_completed(futures), configs
        ):
            try:
                run_results = future.result()

                # Add configuration info to results
                result_entry = {
                    "Vehicle_Density": density,
                    "Penetration_Rate": rate,
                    "Algorithm": algorithm_name,
                    "Run": run,
                    "EAR_Mean": run_results["EAR"]["mean"],
                    "EAR_Median": run_results["EAR"]["median"],
                    "CBR_Mean": run_results["CBR"]["mean"],
                    "CBR_Median": run_results["CBR"]["median"],
                    "AOI_Mean": run_results["AOI"]["mean"],
                    "AOI_Median": run_results["AOI"]["median"],
                    "CPM_Size_Mean": run_results["CPM_Size"]["mean"],
                    "CPM_Size_Median": run_results["CPM_Size"]["median"],
                }

                results.append(result_entry)
                logger.debug(f"Run {run+1} completed")

            except Exception as e:
                logger.error(f"Error in simulation: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    logger.info(f"Experiment completed with {len(results_df)} results")
    return results_df


def plot_comparison_results(results_df):
    """
    Plot comparison results matching those in the Wolff et al. paper with box plots

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing all simulation results
    """
    logger.info("Ploting results...")
    # Set seaborn style
    sns.set(style="whitegrid")

    # Create figure with subplots for standard analysis - matching Figure 3, 4, 5, 6 in paper
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # For EAR vs Penetration Rate plots (Figure 3a & 3b) - Using Box Plots
    # Low density plot (Figure 3a)
    low_density = results_df[results_df["Vehicle_Density"] == 30]
    sns.boxplot(
        data=low_density,
        x="Penetration_Rate",
        y="EAR_Mean",
        hue="Algorithm",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("EAR vs Penetration Rate (Low Density)")
    axes[0, 0].set_xlabel("Penetration Rate [%]")
    axes[0, 0].set_ylabel("Environmental Awareness Ratio")
    axes[0, 0].set_ylim(0, 1.05)  # Match Figure 3 y-axis

    # First set the tick positions, then the labels
    tick_positions = [0, 1, 2, 3]
    axes[0, 0].set_xticks(tick_positions)
    axes[0, 0].set_xticklabels(["5%", "10%", "25%", "50%"])

    # High density plot (Figure 3b)
    high_density = results_df[results_df["Vehicle_Density"] == 60]
    sns.boxplot(
        data=high_density,
        x="Penetration_Rate",
        y="EAR_Mean",
        hue="Algorithm",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("EAR vs Penetration Rate (High Density)")
    axes[0, 1].set_xlabel("Penetration Rate [%]")
    axes[0, 1].set_ylabel("Environmental Awareness Ratio")
    axes[0, 1].set_ylim(0, 1.05)  # Match Figure 3 y-axis

    # First set the tick positions, then the labels
    axes[0, 1].set_xticks(tick_positions)
    axes[0, 1].set_xticklabels(["5%", "10%", "25%", "50%"])

    # Plot CBR vs Penetration Rate for high density - Figure 4
    sns.boxplot(
        data=high_density,
        x="Penetration_Rate",
        y="CBR_Mean",
        hue="Algorithm",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("CBR vs Penetration Rate (High Density)")
    axes[1, 0].set_xlabel("Penetration Rate [%]")
    axes[1, 0].set_ylabel("Channel Busy Ratio")

    # First set the tick positions, then the labels
    axes[1, 0].set_xticks(tick_positions)
    axes[1, 0].set_xticklabels(["5%", "10%", "25%", "50%"])

    # Create CDF plot for Age of Information - Figure 6
    # Get AOI data from results
    aoi_data = []

    for alg in results_df["Algorithm"].unique():
        alg_data = results_df[results_df["Algorithm"] == alg]["AOI_Mean"].tolist()
        if alg_data:
            # Convert to seconds if needed
            aoi_data.append(
                (
                    alg,
                    (
                        np.array(alg_data) / 1000
                        if np.mean(alg_data) > 100
                        else np.array(alg_data)
                    ),
                )
            )

    # Plot CDF
    for alg, data in aoi_data:
        # Sort the data
        data_sorted = np.sort(data)
        # Calculate the CDF
        p = 1.0 * np.arange(len(data)) / (len(data) - 1)
        # Plot the CDF
        axes[1, 1].plot(data_sorted, p, label=alg)

    axes[1, 1].set_title("Age of Information")
    axes[1, 1].set_xlabel("Age of Information [s]")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].legend()

    # Save and show results
    fig.tight_layout()
    fig.savefig("vanet_simulation_results.png", dpi=300)
    plt.show()

    # Create a separate figure for CPM Size (Figure 5)
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Filter for 25% penetration rate as in the paper
    pen_25 = results_df[results_df["Penetration_Rate"] == 0.05]

    # Create boxplot (this part was already correct)
    sns.boxplot(
        data=pen_25, x="Algorithm", y="CPM_Size_Mean", hue="Vehicle_Density", ax=ax
    )

    ax.set_title("Potential Message Size by Algorithm (25% Penetration Rate)")
    ax.set_xlabel("CPS Mode")
    ax.set_ylabel("Potential Message Size [#Objects]")

    # Custom legend to match paper
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Low", "High"], title="Traffic Density")

    # Save this figure too
    fig2.tight_layout()
    fig2.savefig("vanet_cpm_size_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run VANET simulation")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the simulation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with fewer configurations",
    )

    # Add logging arguments
    parser.add_argument(
        "--console-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level",
    )
    parser.add_argument(
        "--file-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="File logging level",
    )
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument(
        "--log-components",
        nargs="+",
        choices=["simulation", "environment", "vehicle", "network", "metrics"],
        help="Set specific components to DEBUG level",
    )
    parser.add_argument("--threads", default=10, type=int)

    args = parser.parse_args()

    # Set up logging with command-line options
    setup_logging(
        console_level=getattr(logging, args.console_level),
        file_level=getattr(logging, args.file_level),
        log_dir=args.log_dir,
    )

    # Set specific components to DEBUG if requested
    if args.log_components:
        for component in args.log_components:
            set_component_level(component, logging.DEBUG)
            logger.info(f"Set {component} logging to DEBUG level")

    logger.info("VANET Simulation starting")

    if args.quick:
        # Quick test run with limited configurations
        logger.info("Running quick test simulation...")
        vehicle_densities = [30]
        penetration_rates = [0.05]
        algorithms = [
            ForwardingAlgorithm.NO_FORWARDING,  # Baseline ETSI CPS
            ForwardingAlgorithm.MULTI_HOP,  # Proposed algorithm
        ]
        num_runs = 2  # Just 2 runs for quick testing
        simulation_time = 10  # Shorter simulation time
    else:
        # Full experiment matching Wolff paper parameters
        logger.info("Running full VANET simulation experiment...")
        vehicle_densities = [30, 60]  # Low and high density as in paper
        penetration_rates = [0.05, 0.1, 0.25, 0.5]  # Match paper's values
        algorithms = [
            ForwardingAlgorithm.NO_FORWARDING,  # Baseline ETSI CPS
            # ForwardingAlgorithm.GBC,  # GBC forwarding
            ForwardingAlgorithm.MULTI_HOP,  # Proposed algorithm
        ]
        num_runs = 10  # 10 runs per configuration as in the paper
        simulation_time = 15  # 15 seconds per run as in the paper

    # Run experiment
    results = run_experiment_parallel(
        vehicle_densities=vehicle_densities,
        penetration_rates=penetration_rates,
        algorithms=algorithms,
        num_runs=num_runs,
        simulation_time=simulation_time,
        visualize=args.visualize,
        max_workers=args.threads,
    )

    # Save results to CSV
    results.to_csv("vanet_simulation_results.csv", index=False)
    logger.info("Results saved to 'vanet_simulation_results.csv'")

    # Plot results
    logger.info("Generating plots...")
    plot_comparison_results(results)
