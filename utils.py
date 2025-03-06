import datetime
import os
import time

# Define a constant for the stop file path
STOP_FILE_PATH = "simulation_stop.flag"


def generate_unique_filename(prefix, extension, folder=None):
    """Generate a unique filename with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.{extension}"

    if folder:
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)
    else:
        return filename


def setup_result_folders():
    """Create folder structure for results"""
    folders = {"base": "results", "csv": "results/csvs", "plots": "results/plots"}

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


# Utility functions to check and set the stop flag
def set_stop_flag():
    """Create a stop file to signal all processes to stop"""
    try:
        with open(STOP_FILE_PATH, "w") as f:
            f.write("STOP")
        # Small delay to ensure other processes can see the file
        time.sleep(0.2)
    except Exception as e:
        print(f"Error setting stop flag: {e}")


def check_stop_flag():
    """Check if the stop file exists"""
    return os.path.exists(STOP_FILE_PATH)


def clear_stop_flag():
    """Remove the stop file if it exists"""
    if os.path.exists(STOP_FILE_PATH):
        try:
            os.remove(STOP_FILE_PATH)
        except Exception as e:
            print(f"Error clearing stop flag: {e}")
