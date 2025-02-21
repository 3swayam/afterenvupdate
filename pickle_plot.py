import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

N_STEPS_PER_EPISODE = 1350  # ✅ Defines number of steps per episode

# ✅ Ensure the plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_metrics(filename):

    with open(filename, 'rb') as f:
        return pickle.load(f)

def moving_average(data, window_size):

    if len(data) < window_size:
        return np.array(data)  # Return original if not enough data points
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_metric(data, label, title, filename):

    if not data:  # ✅ Handle empty data
        print(f"Skipping {title}: No data available.")
        return

    plt.figure()
    plt.plot(moving_average(data, 10), label=label)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(f'plots/{filename}')
    plt.show()

# ✅ Load training metrics
train_metrics = load_metrics('optimized_multi_agent_model_train.pkl')

print("Available keys in train_metrics:", train_metrics.keys())  # Debugging log

# ✅ Plot key metrics
plot_metric(train_metrics.get('reward_history', []), 'Reward', 'Episode vs Reward', 'reward_vs_episode.png')
plot_metric(train_metrics.get('vehicle_energy_history', []), 'Vehicle Energy', 'Episode vs Energy', 'energy_vs_episode.png')
plot_metric(train_metrics.get('offloading_rate_history', []), 'Offloading Rate', 'Episode vs Offloading Rate', 'offloading_rate_vs_episode.png')
plot_metric(train_metrics.get('computation_delay_history', []), 'Computation Delay', 'Episode vs Computation Delay', 'computation_delay_vs_episode.png')
plot_metric(train_metrics.get('n_tasks_executed_history', []), 'Number of Tasks Executed', 'Episode vs Number of Tasks Executed', 'tasks_executed_vs_episode.png')
