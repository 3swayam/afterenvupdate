import time
from Config import Config
from Vehicle import Vehicle
from RSU import RSU
from Env import Env
from Agent import DQNAgent
import csv

states = [[5,5,0,0],[5,3,0,0],[10,10,0,0],[5,2.5,0,0],[10,12,0,0],[10,9,0,0],[5,8,0,0],[10,15,0,0],[5,7,0,0],[10,7,0,0]]

class RealWorld:
    def __init__(self):
        """
        Initializes the real-world simulation environment.
        """
        self.vehicle = Vehicle()
        self.env = Env(self.vehicle,states)
        self.dqn_agent = DQNAgent(
            device="cpu",
            state_size=Config.STATE_SIZE,
            action_size=Config.ACTION_SIZE,
            discount=Config.DISCOUNT,
            memory_capacity=Config.MEMORY_CAPACITY,
            lr=Config.LEARNING_RATE,
            train_mode=False
        )
        self.dqn_agent.load_model(Config.MODEL_NAME)

    def execute_task(self, state):
        episode_states_action = []
        """
        Determines whether to offload a task to an RSU or process it locally.

        Args:
        - task_size (float): The task size in bits.

        Returns:
        - dict: Execution details including time, energy, and decision.
        """
        action = self.dqn_agent.select_action(state)  # AI-based decision making
        task_size = state[0]
        
        if action:
            connected_rsus = self.env.get_connected_servers()
            # Distribute workload to all connected servers
            if connected_rsus:
                task_size = state[0] * 1e6 / len(connected_rsus)
                execution_time =0
                energy_consumption=0
                for rsu in connected_rsus:
                    rsu_comm_delay_single = rsu.commDelay(task_size, self.vehicle.stayTime(rsu.stay_dist), self.vehicle.speed, self.vehicle.power) + Config.LATENCY
                    execution_time += rsu.compDelay(task_size) + rsu_comm_delay_single
                    energy_consumption += rsu.compute_energy(task_size, self.vehicle.stayTime(rsu.stay_dist), self.vehicle.speed,
                                                    self.vehicle.power)
            else:
                print("NO connected RSUS")
            decision = "Offloaded"
        else:  # Process locally
            execution_time = self.vehicle.compDelay(task_size * 1e6)
            energy_consumption = self.vehicle.compute_energy(task_size * 1e6)
            decision = "Processed Locally"

        episode_states_action.append(state + [action])
        
        # Append data instead of overwriting
        file_exists = False
        try:
            with open('RealWorld.csv', 'r') as f:
                file_exists = True
        except FileNotFoundError:
            pass
        
        with open('RealWorld.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if the file does not exist
            if not file_exists:
                writer.writerow(['TaskSize', 'TaskDeadline', 'VehicleLoadFactor', 'RSULoadFactor', 'OffloadDecision'])
            
            for item in episode_states_action:
                writer.writerow(item)

        return {
            "task_size": task_size,
            "execution_time": execution_time,
            "energy_consumption": energy_consumption,
            "decision": decision
        }

    def run_simulation(self, num_tasks):
        """
        Simulates real-world vehicle movement and task execution.

        Args:
        - num_tasks (int): Number of tasks to execute.
        """
        for index in range(num_tasks):
            # task_size = states[index][0]  # Dynamic task generation
            result = self.execute_task(states[index])
            print(f"Task {result['decision']} - Time: {result['execution_time']:.4f}s, Energy: {result['energy_consumption']:.4f}J")
            self.vehicle.move(Config.TIME_STEP)  # Vehicle moves after task execution
            self.env.add_remove_servers()  # Update RSUs dynamically


if __name__ == "__main__":
    realworld = RealWorld()
    realworld.run_simulation(len(states))  # Change: Simulates **10 tasks execution**
