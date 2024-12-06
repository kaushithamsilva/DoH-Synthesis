import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
from itertools import product
from pyts.image import MarkovTransitionField


def ensure_directories(locations):
    """Create figure directories if they don't exist."""
    for location in locations:
        os.makedirs(
            f"../../figures/MarkovTransitionField/{location}", exist_ok=True)


def plot_and_save_trace(args):
    """Plot and save a single trace."""
    trace, location, website, sample = args
    trace_2d = transformer.transform(trace.reshape((-1, len(trace))))[0]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(trace_2d)
    ax.axis('off')  # Turn off axis to remove ticks

    # Save the image without extra borders or padding
    plt.savefig(f"../../figures/MarkovTransitionField/{location}/{website}_{sample}.png",
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    locations = ['LOC1', 'LOC2']

    # Create necessary directories
    ensure_directories(locations)

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv"
    )

    # Group the data
    gp = df.groupby(['Location', 'Website'])

    # 1d to 2d
    transformer = MarkovTransitionField()

    # Create a list of all plotting tasks
    plot_tasks = []
    for (location, website), group in gp:
        for i in range(200):
            plot_tasks.append((group.iloc[i, 2:].values, location, website, i))

    # Use a single pool for all plotting tasks
    print(f"Starting parallel processing with {cpu_count()} processes...")
    with Pool(processes=cpu_count()) as pool:
        pool.map(plot_and_save_trace, plot_tasks)

    print("Processing complete!")
