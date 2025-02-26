import time
from Config import Config
from Vehicle import Vehicle
from RSU import RSU
from Env import Env
from Agent import DQNAgent
import torch

def offloading_scenario():
    v_bf = []
    r_bf = []
    v_lf = 0
    r_lf = 0
    v_size = 40
    r_size = 80
    
    print("Vehicle buffer initially empty")
    print("RSU buffer initially empty")
    
    vehicle = Vehicle()
    
    dqn_agent = DQNAgent(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        state_size=Config.STATE_SIZE,
            action_size=Config.ACTION_SIZE,
            discount=Config.DISCOUNT,
            memory_capacity=50000,  # Increased memory
            lr=5e-4,  # Lowered learning rate
            eps_max=0.5,  # Lowered initial epsilon
            eps_decay=0.997,  # Faster decay
            train_mode=False)
    
    
    # Load pre-trained model
    dqn_agent.load_model(Config.MODEL_NAME)
    
    tasks = [[5,5,0,0],[5,3,0,0],[10,10,0,0],[5,2.5,0,0],[10,12,0,0],[10,9,0,0],[5,8,0,0],[10,15,0,0],[5,7,0,0],[10,7,0,0]]
    for i, state in enumerate(tasks):
        print("#####################################")
        print(f"Task {i+1}:")
        print(f"Task size: {state[0]}")
        print(f"Task Deadline: {state[1]}")
        print("Task values have been given to Task Offloading Decision Model")
        
        # Create a state based on task size, deadline, vehicle load factor, and RSU load factor
        action = dqn_agent.select_action(state)
        
        if action == 0:
            v_bf.append(state[0])
            v_lf = (sum(v_bf) / v_size) * 100
            if v_lf > 100:
                print(f"This task is done at local but it overloads the vehicle by {v_lf - 100}%.")
                # v_lf -= 100
            else:
                print("This task is done at local")
        else:
            env = Env(vehicle, tasks, False)
            rsu = env.selectClosestServer()
            stayTime = vehicle.stayTime(rsu.stay_dist)
            r_bf.append(state[0])
            r_lf = (sum(r_bf) / r_size) * 100
            if r_lf > 100:
                print(f"This task has been offloaded but it overloads the RSU by {r_lf - 100}%.")
                # r_lf -= 100
            else:
                print("This task has been offloaded")
        
        print("Vehicle's Buffer:", v_bf)
        print(f"Vehicle's Loadfactor: {v_lf:.2f}%.")
        print("RSU's Buffer:", r_bf)
        print(f"RSU's Loadfactor: {r_lf:.2f}%")
        
        # Checkpoint to exit if both load factors reach 100%
        if v_lf >= 100 and r_lf >= 100:
            print("Both Vehicle and RSU load factors have reached 100%. Exiting...")
            break
        
        
        stayTime -= state[1] / 2
        print("Remaining Stay Time of the vehicle", stayTime)
        rsu_delay = rsu.compDelay(state[0] * 1e6) + rsu.commDelay(state[0] * 1e6, stayTime, vehicle.speed, vehicle.power)
        
        if stayTime < rsu_delay and action == 1:
            print("Remaining Stay Time of the vehicle is lesser than the execution time. So Results cannot be returned to Vehicle by Current RSU")
            break

if __name__ == '__main__':
    offloading_scenario()
