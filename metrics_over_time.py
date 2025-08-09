import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def extract_all_metrics_from_logfile(path):
    with open(path, 'r') as f:
        content = f.read()

    # Find all JSON-like blocks using regex
    blocks = re.findall(r"\{.*?\}", content, re.DOTALL)

    parsed = []
    for block in blocks:
        try:
            metrics = json.loads(block.replace("'", '"'))
            parsed.append(metrics)
        except json.JSONDecodeError:
            continue  # skip malformed blocks

    parsed = [e for e in parsed if 'loc_error@10lines' in e]

    return parsed

def plot_metrics(name, metrics_list):
    epochs = np.arange(len(metrics_list))

    def extract_array(key):
        return np.array([m.get(key, np.nan) for m in metrics_list])

    # Plot repeatability
    plt.figure()
    for px in [1, 3, 5]:
        y = extract_array(f'repeatability@{px}px')
        plt.plot(epochs, y, label=f'@{px}px')
    plt.title("Repeatability over time")
    plt.xlabel("Run index")
    plt.ylabel("Repeatability")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_repeatability_plot.png", dpi=300)

    # Plot localization error
    plt.figure()
    for lines in [10, 50, 300]:
        y = extract_array(f'loc_error@{lines}lines')
        plt.plot(epochs, y, label=f'@{lines} lines')
    plt.plot(epochs, extract_array('mloc_error'), '--', label='mloc_error')
    plt.title("Localization Error over time")
    plt.xlabel("Run index")
    plt.ylabel("Localization Error")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_loc_error_plot.png", dpi=300)

    # Plot homography error
    plt.figure()
    for k in [1, 3, 5]:
        y = extract_array(f'mH_err@{k}')
        plt.plot(epochs, y, label=f'mH_err@{k}')
    plt.title("Homography Error over time")
    plt.xlabel("Run index")
    plt.ylabel("Homography Error")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_homography_plot.png", dpi=300)

# === Main ===
if __name__ == "__main__":

    def extract_experiment_name(path):
        with open(path, 'r') as f:
            content = f.read()
            matches = re.findall(r"outputs/training/([^/]+)/checkpoint", content)
            return matches[0]  # list of all matched experiment names

    log_path = "slurm-39561514.out"  # change to your path

    metrics = extract_all_metrics_from_logfile(log_path)
    if not metrics:
        print("No metrics found.")
    else:
        plot_metrics(extract_experiment_name(log_path), metrics)
