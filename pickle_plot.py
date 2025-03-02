import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from Config import Config

N_STEPS_PER_EPISODE = 1350  # Defines number of steps per episode
train_mode = Config.TRAIN_MODE
model_filename = Config.MODEL_NAME
folder = "train" if train_mode else "test"

def load_metrics(filename):
    with open(filename, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

# Function to combine and average energy values
def combine_and_average_energy(metrics):
    combined_energy_history = []
    for vehicle_energy, rsu_energy in zip(metrics.get('vehicle_energy_history', []), metrics.get('rsu_energy_history', [])):
        combined_energy = (vehicle_energy + rsu_energy)
        combined_energy_history.append(combined_energy)
    return combined_energy_history

# Function to combine and average computation delay values
def combine_and_average_comp_delay(metrics):
    combined_comp_delay_history = []
    for vehicle_comp_delay, rsu_comp_delay in zip(metrics.get('vehicle_comp_delay_history', []), metrics.get('rsu_comp_delay_history', [])):
        combined_comp_delay = (vehicle_comp_delay + rsu_comp_delay)
        combined_comp_delay_history.append(combined_comp_delay)
    return combined_comp_delay_history

# Function to calculate offloading rate
def calculate_offloading_rate(metrics):
    offloading_rate_history = []
    for n_rsu_tasks in metrics.get('n_rsu_tasks_history', []):
        offloading_rate = n_rsu_tasks / N_STEPS_PER_EPISODE
        offloading_rate_history.append(offloading_rate)
    return offloading_rate_history

# Function to smooth data using a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Ensure the plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load training metrics
if train_mode:
    file_name= f"{model_filename}_train.pkl"
else:
    file_name= f"{model_filename}_test.pkl"
train_metrics = load_metrics(file_name)

print("Available keys in train_metrics:", train_metrics.keys())  # Debugging log

# Combine and average energy values
combined_energy_history = combine_and_average_energy(train_metrics)

# Combine and average computation delay values
combined_comp_delay_history = combine_and_average_comp_delay(train_metrics)

# Calculate offloading rate
offloading_rate_history = calculate_offloading_rate(train_metrics)

# Add combined energy, computation delay history, and offloading rate to metrics
train_metrics['combined_energy_history'] = combined_energy_history
train_metrics['combined_comp_delay_history'] = combined_comp_delay_history
train_metrics['offloading_rate_history'] = offloading_rate_history

# Set smoothing window size
window_size = 10 if train_mode else 1

# Plot Reward vs Episode
plt.figure()
plt.plot(moving_average(train_metrics.get('reward_history', []), window_size), label='Reward')
plt.title('Episode vs Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig(f"plots/{folder}/reward_vs_episode.png")
plt.show()

# Plot Energy vs Episode
plt.figure()
plt.plot(moving_average(train_metrics.get('vehicle_energy_history', []), window_size), label='Vehicle Energy')
plt.plot(moving_average(train_metrics.get('rsu_energy_history', []), window_size), label='RSU Energy')
plt.plot(moving_average(train_metrics.get('combined_energy_history', []), window_size), label='Combined Energy')
plt.title('Episode vs Energy')
plt.xlabel('Episode')
plt.ylabel('Energy')
plt.legend()
plt.savefig(f"plots/{folder}/energy_vs_episode.png")
plt.show()

# Plot Computation Delay vs Episode
plt.figure()
plt.plot(moving_average(train_metrics.get('vehicle_comp_delay_history', []), window_size), label='Vehicle Comp Time')
plt.plot(moving_average(train_metrics.get('rsu_comp_delay_history', []), window_size), label='RSU Comp Time')
plt.plot(moving_average(train_metrics.get('combined_comp_delay_history', []), window_size), label='Combined Comp Time')
plt.title('Episode vs Computation Time')
plt.xlabel('Episode')
plt.ylabel('Computation Time')
plt.legend()
plt.savefig(f"plots/{folder}/comp_delay_vs_episode.png")
plt.show()

# # Plot Offloading Rate vs Episode
plt.figure()
plt.plot(moving_average(train_metrics.get('offloading_rate_history', []), window_size), label='Offloading Rate')
plt.title('Episode vs Offloading Rate')
plt.xlabel('Episode')
plt.ylabel('Offloading Rate')
plt.legend()
plt.savefig(f"plots/{folder}/offloading_rate_vs_episode.png")
plt.show()

# # Plot Number of Tasks Executed vs Episode
plt.figure()
plt.plot(moving_average(train_metrics.get('n_vehicle_tasks_history', []), window_size), label='Number of Tasks Executed in Vehicle')
plt.plot(moving_average(train_metrics.get('n_rsu_tasks_history', []), window_size), label='Number of Tasks Executed in RSU')
plt.title('Episode vs Number of Tasks Executed')
plt.xlabel('Episode')
plt.ylabel('Number of Tasks Executed')
plt.legend()
plt.savefig(f"plots/{folder}/n_tasks_vs_episode.png")
plt.show()
