import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from framework import ForwardingAlgorithm, Simulation


def run_experiment(
    vehicle_densities,
    penetration_rates,
    algorithms,
    num_runs=10,
    simulation_time=15,
    visualize=False,
):
    """
    Run simulations for different configurations and collect results

    Parameters:
    -----------
    vehicle_densities : list
        List of vehicle densities to test (vehicles/km)
    penetration_rates : list
        List of penetration rates to test (percentage as decimal)
    algorithms : list
        List of forwarding algorithms to test
    num_runs : int
        Number of simulation runs for each configuration
    simulation_time : float
        Duration of each simulation run in seconds
    visualize : bool
        Whether to visualize the simulation

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing all simulation results
    """
    results = []

    # Total configurations
    total_configs = (
        len(vehicle_densities) * len(penetration_rates) * len(algorithms) * num_runs
    )

    with tqdm(total=total_configs) as pbar:
        for density in vehicle_densities:
            for rate in penetration_rates:
                for algorithm in algorithms:
                    algorithm_name = algorithm.name

                    for run in range(num_runs):
                        config = {
                            "vehicle_density": density,
                            "penetration_rate": rate,
                            "algorithm": algorithm,
                        }

                        # Create and run simulation
                        sim = Simulation(config)

                        sim.run(
                            simulation_time=simulation_time,
                            visualize=visualize,
                            dt=0.1,
                        )

                        # Get results
                        run_results = sim.get_results()

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
                        pbar.update(1)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def plot_comparison_results(results_df):
    """
    Plot comparison results similar to those in the Wolff et al. paper

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing all simulation results
    """
    # Set seaborn style
    sns.set(style="whitegrid")

    # Create figure with subplots for standard analysis - matching Figure 3, 4, 5, 6 in paper
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Group results by algorithm, density, and penetration rate
    grouped_results = (
        results_df.groupby(["Algorithm", "Vehicle_Density", "Penetration_Rate"])
        .mean()
        .reset_index()
    )

    # Plot EAR vs Penetration Rate for different algorithms (Low Density) - Figure 3a
    low_density = grouped_results[grouped_results["Vehicle_Density"] == 30]
    sns.lineplot(
        data=low_density,
        x="Penetration_Rate",
        y="EAR_Mean",
        hue="Algorithm",
        marker="o",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("EAR vs Penetration Rate (Low Density)")
    axes[0, 0].set_xlabel("Penetration Rate [%]")
    axes[0, 0].set_ylabel("Environmental Awareness Ratio")
    axes[0, 0].set_ylim(0.5, 1.05)  # Match Figure 3 y-axis

    # Convert decimal to percentage for x-axis ticks
    axes[0, 0].set_xticks([0.05, 0.1, 0.25, 0.5])
    axes[0, 0].set_xticklabels(["05", "10", "25", "50"])

    # Plot EAR vs Penetration Rate for different algorithms (High Density) - Figure 3b
    high_density = grouped_results[grouped_results["Vehicle_Density"] == 60]
    sns.lineplot(
        data=high_density,
        x="Penetration_Rate",
        y="EAR_Mean",
        hue="Algorithm",
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("EAR vs Penetration Rate (High Density)")
    axes[0, 1].set_xlabel("Penetration Rate [%]")
    axes[0, 1].set_ylabel("Environmental Awareness Ratio")
    axes[0, 1].set_ylim(0.5, 1.05)  # Match Figure 3 y-axis

    # Convert decimal to percentage for x-axis ticks
    axes[0, 1].set_xticks([0.05, 0.1, 0.25, 0.5])
    axes[0, 1].set_xticklabels(["05", "10", "25", "50"])

    # Plot CBR vs Penetration Rate for high density - Figure 4
    sns.lineplot(
        data=high_density,
        x="Penetration_Rate",
        y="CBR_Mean",
        hue="Algorithm",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("CBR vs Penetration Rate (High Density)")
    axes[1, 0].set_xlabel("Penetration Rate [%]")
    axes[1, 0].set_ylabel("Channel Busy Ratio")
    axes[1, 0].set_ylim(0, 0.6)  # Match Figure 4 y-axis

    # Convert decimal to percentage for x-axis ticks
    axes[1, 0].set_xticks([0.05, 0.1, 0.25, 0.5])
    axes[1, 0].set_xticklabels(["05", "10", "25", "50"])

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
    axes[1, 1].set_xlim(0, 1.0)  # Match Figure 6 x-axis
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].legend()

    # Save and show results
    fig.tight_layout()
    fig.savefig("vanet_simulation_results.png", dpi=300)
    plt.show()

    # Create a separate figure for CPM Size (Figure 5)
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Filter for 25% penetration rate as in the paper
    pen_25 = grouped_results[grouped_results["Penetration_Rate"] == 0.25]

    # Create boxplot
    sns.boxplot(
        data=pen_25, x="Algorithm", y="CPM_Size_Mean", hue="Vehicle_Density", ax=ax
    )

    ax.set_title("Potential Message Size by Algorithm (25% Penetration Rate)")
    ax.set_xlabel("CPS Mode")
    ax.set_ylabel("Potential Message Size [#Objects]")
    ax.set_ylim(0, 200)  # Match Figure 5 y-axis

    # Custom legend to match paper
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Low", "High"], title="Traffic Density")

    # Save this figure too
    fig2.tight_layout()
    fig2.savefig("vanet_cpm_size_comparison.png", dpi=300)


if __name__ == "__main__":
    # Add this import if not already present
    import argparse

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
    args = parser.parse_args()

    if args.quick:
        # Quick test run with limited configurations
        print("Running quick test simulation...")
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
        print("Running full VANET simulation experiment...")
        vehicle_densities = [30, 60]  # Low and high density as in paper
        penetration_rates = [0.05, 0.1, 0.25, 0.5]  # Match paper's values
        algorithms = [
            ForwardingAlgorithm.NO_FORWARDING,  # Baseline ETSI CPS
            ForwardingAlgorithm.GBC,  # GBC forwarding
            ForwardingAlgorithm.MULTI_HOP,  # Proposed algorithm
        ]
        num_runs = 10  # 10 runs per configuration as in the paper
        simulation_time = 15  # 15 seconds per run as in the paper

    # Run experiment
    results = run_experiment(
        vehicle_densities=vehicle_densities,
        penetration_rates=penetration_rates,
        algorithms=algorithms,
        num_runs=num_runs,
        simulation_time=simulation_time,
        visualize=args.visualize,
    )

    # Save results to CSV
    results.to_csv("vanet_simulation_results.csv", index=False)
    print("Results saved to 'vanet_simulation_results.csv'")

    # Plot results
    print("Generating plots...")
    plot_comparison_results(results)
    print(
        "Plots saved to 'vanet_simulation_results.png' and 'vanet_cpm_size_comparison.png'"
    )
