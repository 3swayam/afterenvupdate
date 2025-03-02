import torch
import pickle
import numpy as np
from tqdm import tqdm
from Config import Config
from Vehicle import Vehicle
from Env import Env
from Agent import DQNAgent
import csv
from Dataset import genStates 


def train(env, dqn_agent, num_train_eps, update_frequency, batch_size, model_filename):
    """
    Trains the RL model with multiple RSUs and real-time vehicle movement.
    """
    reward_history = []
    vehicle_energy_history = []
    vehicle_comp_delay_history = []
    rsu_comp_delay_history = []
    rsu_energy_history=[]
    n_vehicle_tasks_history = []
    n_rsu_tasks_history = []

    min_memory_size = batch_size * 5  # Ensure memory is filled before training
    step_cnt = 0
    best_score = -np.inf
    
    for ep_cnt in tqdm(range(num_train_eps), desc="Training Progress", unit="episode"):
        
        done = False
        state = env.reset()
        ep_score = 0
        vehicle_energy = 0
        rsu_energy = 0
        vehicle_comp_delay = 0
        rsu_comp_delay = 0
        n_vehicle_tasks = 0
        n_rsu_tasks = 0
        
        while not done:
            action = dqn_agent.select_action(state)
            # next_state, reward, done = env.step(state,action)
            next_state, reward, done, connected_rsus = env.step(state, action) # Get connected RSUs

            dqn_agent.memory.store(state, action, next_state, reward, done)
            if len(dqn_agent.memory) > min_memory_size:
                dqn_agent.learn(batch_size=batch_size)
            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state

            ep_score += reward
            if action:
                # Distribute workload to all connected servers
                if connected_rsus:
                    task_size = state[0] * 1e6 / len(connected_rsus)
                    for rsu in connected_rsus:
                        rsu_energy += rsu.compute_energy(task_size, env.vehicle.stayTime(rsu.stay_dist), env.vehicle.speed, env.vehicle.power)
                        rsu_comm_delay_single = rsu.commDelay(task_size, env.vehicle.stayTime(rsu.stay_dist), env.vehicle.speed, env.vehicle.power) + Config.LATENCY
                        rsu_comp_delay += rsu.compDelay(task_size) + rsu_comm_delay_single
                        n_rsu_tasks += 1
                else:
                    print("NO connected RSUS")
            else:
                vehicle_energy += env.vehicle.compute_energy(state[0]*1e6)
                vehicle_comp_delay += env.vehicle.compDelay(state[0]*1e6)
                n_vehicle_tasks += 1  

            step_cnt += 1 
            

        dqn_agent.update_epsilon()
        # ADDING NEW CODE
        reward_history.append(ep_score)
        vehicle_energy_history.append(vehicle_energy)
        rsu_energy_history.append(rsu_energy)
        vehicle_comp_delay_history.append(vehicle_comp_delay)
        rsu_comp_delay_history.append(rsu_comp_delay)
        n_vehicle_tasks_history.append(n_vehicle_tasks)
        n_rsu_tasks_history.append(n_rsu_tasks)
    

        # END OF NEW CODE

        # Save model if improvement is detected
        if ep_score >= max(reward_history[-10:], default=-np.inf):
            dqn_agent.save_model(model_filename)

    # Save all tracked metrics
    with open(f'{model_filename}_train.pkl', 'wb') as f:
        pickle.dump({
            'reward_history': reward_history,
            'vehicle_energy_history': vehicle_energy_history,
            'rsu_energy_history':rsu_energy_history,
            'vehicle_comp_delay_history':vehicle_comp_delay_history,
            'rsu_comp_delay_history':rsu_comp_delay_history,
            'n_vehicle_tasks_history': n_vehicle_tasks_history,
            'n_rsu_tasks_history': n_rsu_tasks_history,
        }, f)

def test(env, dqn_agent, num_test_eps):
    step_cnt = 0
    reward_history = []
    vehicle_energy_history = []
    rsu_energy_history = []
    vehicle_comp_delay_history = []
    rsu_comp_delay_history = []
    n_vehicle_tasks_history = []
    n_rsu_tasks_history = []

    for ep in range(num_test_eps):
        score = 0
        vehicle_energy = 0
        rsu_energy = 0
        vehicle_comp_delay = 0
        rsu_comp_delay = 0
        done = False
        state = env.reset()
        episode_states_action = []
        n_vehicle_tasks = 0
        n_rsu_tasks = 0

        while not done:
            action = dqn_agent.select_action(state)
            # print(state, action)
            # next_state, reward, done = env.step(state,action)
            next_state, reward, done, connected_rsus = env.step(state, action) # Get connected RSUs
            score += reward

            if action:
                # Distribute workload to all connected servers
                if connected_rsus:
                    task_size = state[0] * 1e6 / len(connected_rsus)
                    for rsu in connected_rsus:
                        rsu_energy += rsu.compute_energy(task_size, env.vehicle.stayTime(rsu.stay_dist), env.vehicle.speed, env.vehicle.power)
                        rsu_comm_delay_single = rsu.commDelay(task_size, env.vehicle.stayTime(rsu.stay_dist), env.vehicle.speed, env.vehicle.power) + Config.LATENCY
                        rsu_comp_delay += rsu.compDelay(task_size) + rsu_comm_delay_single
                        n_rsu_tasks += 1
                else:
                    print("NO connected RSUS")
            else:
                vehicle_energy += env.vehicle.compute_energy(state[0]*1e6)
                vehicle_comp_delay += env.vehicle.compDelay(state[0]*1e6)
                n_vehicle_tasks += 1

            episode_states_action.append(state + [action])
            state = next_state
            step_cnt += 1

            if ep == num_test_eps - 1:
                with open(f'{model_filename}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    writer.writerow(['TaskSize', 'TaskDeadline', 'VehicleLoadFactor', 'RSULoadFactor', 'OffloadDecision'])
                    for item in episode_states_action:
                        writer.writerow(item)

        reward_history.append(score)
        vehicle_energy_history.append(vehicle_energy)
        rsu_energy_history.append(rsu_energy)
        vehicle_comp_delay_history.append(vehicle_comp_delay)
        rsu_comp_delay_history.append(rsu_comp_delay)
        n_vehicle_tasks_history.append(n_vehicle_tasks)
        n_rsu_tasks_history.append(n_rsu_tasks)
        print('Ep: {}, Score: {}'.format(ep, score))

        # Save all tracked metrics
    with open(f'{model_filename}_test.pkl', 'wb') as f:
        pickle.dump({
            'reward_history': reward_history,
            'vehicle_energy_history': vehicle_energy_history,
            'rsu_energy_history':rsu_energy_history,
            'vehicle_comp_delay_history':vehicle_comp_delay_history,
            'rsu_comp_delay_history':rsu_comp_delay_history,
            'n_vehicle_tasks_history': n_vehicle_tasks_history,
            'n_rsu_tasks_history': n_rsu_tasks_history,
        }, f)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    vehicle = Vehicle()
    train_mode = Config.TRAIN_MODE
    train_states, test_states = genStates()
    env = Env(vehicle, train_states+test_states)
    model_filename = Config.MODEL_NAME

    # new code
    if train_mode:
        train_env = Env(vehicle, train_states,train_mode)
        dqn_agent = DQNAgent(
            device,
            state_size=Config.STATE_SIZE,
            action_size=Config.ACTION_SIZE,
            discount=Config.DISCOUNT,
            memory_capacity=50000,  # Increased memory
            lr=5e-4,  # Lowered learning rate
            eps_max=0.5,  # Lowered initial epsilon
            eps_decay=0.997,  # Faster decay
            train_mode=train_mode)

        train(
            env=train_env,
            dqn_agent=dqn_agent,
            num_train_eps=Config.NUM_TRAIN_EPS,
            update_frequency=Config.UPDATE_FREQUENCY,
            batch_size=Config.BATCH_SIZE,
            model_filename=model_filename
        )

    else:
        test_env = Env(vehicle, test_states,train_mode)
        dqn_agent = DQNAgent(
            device,
            state_size=Config.STATE_SIZE,
            action_size=Config.ACTION_SIZE,
            discount=Config.DISCOUNT,
            memory_capacity=50000,  # Increased memory
            lr=5e-4,  # Lowered learning rate
            eps_max=0.5,  # Lowered initial epsilon
            eps_decay=0.997,  # Faster decay
            train_mode=train_mode)
        dqn_agent.load_model(model_filename)
        test(test_env, dqn_agent, num_test_eps=Config.NUM_TEST_EPS)
    # new code end
