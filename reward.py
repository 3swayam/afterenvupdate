import numpy as np
 
def reward_calculate(action, state, vehicle, rsu, alpha=10, beta=5, gamma=2):
    # 2,1,0.5
    
    # Extract parameters from state
    task_size = state[0] * 1e6  # Convert MB to bits
    task_deadline = state[1]
 
    # Compute **Latency (L)**
    local_processing_delay = vehicle.compDelay(task_size)
    rsu_processing_delay = rsu.compDelay(task_size)
    communication_delay = rsu.commDelay(task_size, vehicle.stayTime(rsu.stay_dist), vehicle.speed, vehicle.power)
 
    latency = (1 - action) * local_processing_delay + action * (rsu_processing_delay + communication_delay)
 
    # Compute **Energy Consumption (E)**
    vehicle_energy = vehicle.compute_energy(task_size)
    rsu_energy = rsu.compute_energy(task_size, vehicle.stayTime(rsu.stay_dist), vehicle.speed, vehicle.power)
 
    energy = (1 - action) * vehicle_energy + action * rsu_energy
 
    # Compute **Cost (C)**
    transmission_cost = action * (task_size / rsu.bandwidth) * rsu.power  # Bandwidth usage cost
    processing_cost = action * (rsu_processing_delay * rsu.power)  # RSU processing cost
 
    cost = transmission_cost + processing_cost
 
    # Compute **Immediate Reward**
    immediate_reward = - (alpha * latency + beta * energy + gamma * cost)
 
    # Apply **Penalties**
    if task_deadline < latency:
        immediate_reward -= 5  # Penalty if deadline is missed
    if vehicle.loadfactor > 8 and action == 0:
        immediate_reward -= 5  # Avoid overloading the vehicle
    if rsu.loadfactor > 8 and action == 1:
        immediate_reward -= 5  # Avoid overloading RSU
    if vehicle.speed > 30 and action == 1:
        immediate_reward -= 2  # If vehicle moves too fast, offloading may fail
 
    # Apply **Incentives**
    if vehicle.loadfactor < 2 and action == 0:
        immediate_reward += 15  # Encourage local execution if vehicle is free
    if rsu.loadfactor < 2 and action == 1:
        immediate_reward += 15  # Encourage offloading if RSU is free

    # if state[5] > 3:
    immediate_reward += 5  # If multiple RSUs are available, offloading is better
 
    return immediate_reward,cost,latency  # Removed Future Reward Component (γ max Q(s', a'))