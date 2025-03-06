# VANET Multi-Hop Collective Perception Simulation

This repository contains a Python simulation framework for evaluating multi-hop collective perception in Vehicular Ad-hoc Networks (VANETs). The simulation replicates the research presented in the paper "An Application Layer Multi-Hop Collective Perception Service for Vehicular Adhoc Networks" by Wolff et al.

## Overview

This simulation evaluates different forwarding algorithms for the Collective Perception Service (CPS) in VANETs, with a focus on improving environmental awareness in scenarios with low market penetration rates. The simulation includes:

- A realistic Manhattan grid environment with buildings and roads
- Vehicle movement simulation with realistic parameters
- Implementation of two CPS modes:
  - Standard ETSI CPS (no forwarding)
  - Application layer multi-hop forwarding (proposed algorithm)
- Metrics collection for:
  - Environmental Awareness Ratio (EAR)
  - Channel Busy Ratio (CBR)
  - Age of Information (AOI)
  - CPM message sizes

## Prerequisites

- Python 3.8+ (tested on 3.9 and 3.12)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

## Installation

```bash
# Clone the repository
git clone git@github.com:comejv/cps_multihop.git
cd cps_multihop

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

## Project Structure

```
vanet-collective-perception/
├── algenum.py           # Enumerations for algorithms
├── environment.py       # Manhattan grid environment
├── log_config.py        # Logging configuration
├── main.py              # Main simulation runner
├── metrics.py           # Metrics collection and analysis
├── network.py           # Wireless network simulation
├── simulation.py        # Core simulation logic
├── vehicle.py           # Vehicle class with CPS implementation
├── results/             # Directory for simulation results
│   ├── csvs/            # CSV result files
│   └── plots/           # Generated plots
└── logs/                # Simulation logs
```

## Usage

### Basic Run

```bash
python main.py
```

This will run the full simulation with default parameters matching those used in the Wolff et al. paper.

> [!NOTE]
> Default configuration will use as many threads as there are cores in your CPU. To change this use the option `--threads N`.

### Quick Test

```bash
python main.py --quick
```

Runs a shorter simulation with fewer configurations for testing purposes.

### Visualization

```bash
python main.py --visualize
```

Enables real-time visualization of the simulation.

> [!WARNING]
> When using real time visualization it is recommended to run only one thread with the option `--threads 1`.

### Additional Options

```
--console-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                      Console logging level (default: INFO)
--file-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                      File logging level (default: WARNING)
--log-dir LOG_DIR     Directory for log files
--log-components {simulation,environment,vehicle,network,metrics}
                      Set specific components to DEBUG level
--threads THREADS     Number of threads for parallel execution
--num-runs NUM_RUNS   Number of simulation runs per configuration
--experiment-tag TAG  Tag to add to output filenames
--load-csv CSV_FILE   Load results from CSV file instead of running simulations
```

## Implemented Algorithms

1. **NO_FORWARDING**: Baseline ETSI CPS implementation without any forwarding mechanism.
2. **MULTI_HOP**: Application layer multi-hop forwarding as proposed in the Wolff et al. paper.

## Example Output

The simulation generates:
- CSV files with all metrics data
- Box plots comparing Environmental Awareness Ratio (EAR) for different algorithms
- Channel Busy Ratio (CBR) analysis
- Age of Information (AOI) cumulative distribution function plots
- CPM message size comparisons

## Extending the Simulation

To add a new forwarding algorithm:
1. Add a new enum value in `algenum.py`
2. Implement the algorithm logic in `vehicle.py` within the `run_cps_algorithm` method

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code for academic research, please cite both the original paper and this implementation:

```
V. A. Wolff, E. Xhoxhi and F. Tautz, "An Application Layer Multi-Hop Collective Perception Service for Vehicular Adhoc Networks," 2024 IEEE Intelligent Vehicles Symposium (IV), Jeju Island, Korea, Republic of, 2024, pp. 1847-1852, doi: 10.1109/IV55156.2024.10588398.

VINCENT, Côme. (2024). VANET Multi-Hop Collective Perception Simulation. https://github.com/comejv/cps_multihop
```

> [!CAUTION]
> This is an independent implementation aimed at replicating the results presented in the Wolff et al. paper. It is not affiliated with or endorsed by the original authors.

## Acknowledgment

This simulation framework is based on the research presented by Wolff et al. in their paper "An Application Layer Multi-Hop Collective Perception Service for Vehicular Adhoc Networks".
