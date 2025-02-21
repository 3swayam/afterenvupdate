import torch
import random
import time
import pickle
import numpy as np
from tqdm import tqdm
from Config import Config
from Vehicle import Vehicle
from RSU import RSU
from Env import Env
from Agent import DQNAgent
import csv


def train(env, dqn_agent, num_train_eps, update_frequency, batch_size, model_filename):
    """
    Trains the RL model with multiple RSUs and real-time vehicle movement.
    """
    reward_history = []
    vehicle_energy_history = []
    offloading_rate_history = []
    computation_delay_history = []
    n_tasks_executed_history = []

    min_memory_size = batch_size * 5  # Ensure memory is filled before training

    for ep_cnt in tqdm(range(num_train_eps), desc="Training Progress", unit="episode"):
        state = env.vehicle.get_state()
        done = False
        ep_score = 0
        step_count = 0  # Track steps per episode

        while not done:
            action = dqn_agent.select_action(state)
            next_state, reward, done = env.step(action)
            dqn_agent.memory.store(state, action, next_state, reward, done)

            # ✅ Track vehicle energy consumption
            if hasattr(env.vehicle, 'compute_energy'):
                vehicle_energy_history.append(
                    env.vehicle.compute_energy(task_size=5))  # Adjust task_size logic if needed

            # ✅ Track computation delay
            if hasattr(env.vehicle, 'compDelay'):
                computation_delay_history.append(env.vehicle.compDelay(task_size=5))

            # ✅ Track number of tasks executed
            if hasattr(env, 'get_tasks_executed'):
                n_tasks_executed_history.append(env.get_tasks_executed())

            # ✅ Track offloading rate (tasks handled by RSU)
            if hasattr(env, 'get_rsu_tasks_handled'):
                offloading_rate_history.append(
                    env.get_rsu_tasks_handled() / max(step_count, 1))  # Avoid division by zero

            # Start learning only after memory is filled
            if len(dqn_agent.memory) > min_memory_size:
                dqn_agent.learn(batch_size=batch_size)

            ep_score += reward
            state = next_state
            step_count += 1

            if step_count > 1000:  # 🚨 Prevent Infinite Loops
                done = True

        dqn_agent.update_epsilon()
        reward_history.append(ep_score)

        # Save model if improvement is detected
        if ep_score >= max(reward_history[-10:], default=-np.inf):
            dqn_agent.save_model(model_filename)

    # ✅ Save all tracked metrics
    with open(f'{model_filename}_train.pkl', 'wb') as f:
        pickle.dump({
            'reward_history': reward_history,
            'vehicle_energy_history': vehicle_energy_history,
            'offloading_rate_history': offloading_rate_history,
            'computation_delay_history': computation_delay_history,
            'n_tasks_executed_history': n_tasks_executed_history
        }, f)


def test(env, dqn_agent, num_test_eps):
    """
    Tests the trained RL model.
    """
    for ep in range(num_test_eps):
        state = env.vehicle.get_state()
        done = False
        score = 0

        while not done:
            action = dqn_agent.select_action(state)
            state, reward, done = env.step(action)
            score += reward

        print(f"Episode {ep}: Score = {score}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    vehicle = Vehicle()
    env = Env(vehicle)
    model_filename = Config.MODEL_NAME

    dqn_agent = DQNAgent(
        device,
        state_size=Config.STATE_SIZE,
        action_size=Config.ACTION_SIZE,
        discount=Config.DISCOUNT,
        memory_capacity=50000,  # Increased memory
        lr=5e-4,  # Lowered learning rate
        eps_max=0.5,  # Lowered initial epsilon
        eps_decay=0.997,  # Faster decay
        train_mode=True
    )

    print("Checking if save_model exists:", hasattr(dqn_agent, "save_model"))

    train(
        env=env,
        dqn_agent=dqn_agent,
        num_train_eps=Config.NUM_TRAIN_EPS,
        update_frequency=Config.UPDATE_FREQUENCY,
        batch_size=Config.BATCH_SIZE,
        model_filename=model_filename
    )

    test(env, dqn_agent, num_test_eps=Config.NUM_TEST_EPS)
