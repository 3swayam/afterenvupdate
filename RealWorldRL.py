import time
from Config import Config
from Vehicle import Vehicle
from RSU import RSU
from Env import Env
from Agent import DQNAgent

class RealWorld:
    def __init__(self):
        """
        Initializes the real-world simulation environment.
        """
        self.vehicle = Vehicle()
        self.env = Env(self.vehicle)
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

    def execute_task(self, task_size):
        """
        Determines whether to offload a task to an RSU or process it locally.

        Args:
        - task_size (float): The task size in bits.

        Returns:
        - dict: Execution details including time, energy, and decision.
        """
        state = self.vehicle.get_state()
        action = self.dqn_agent.select_action(state)  # ✅ AI-based decision making

        if action == 1:  # Offload task to RSU
            rsu = self.env.selectClosestServer()
            execution_time = rsu.compDelay(task_size)
            energy_consumption = rsu.computeEnergyConsumption(task_size)
            decision = "Offloaded"
        else:  # Process locally
            execution_time = self.vehicle.compDelay(task_size)
            energy_consumption = self.vehicle.compute_energy(task_size)
            decision = "Processed Locally"

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
        for _ in range(num_tasks):
            task_size = Config.VEHICLE_FREQUENCY * Config.VEHICLE_CPI * Config.TIME_STEP  # ✅ Dynamic task generation
            result = self.execute_task(task_size)
            print(f"Task {result['decision']} - Time: {result['execution_time']:.4f}s, Energy: {result['energy_consumption']:.4f}J")
            self.vehicle.move(Config.TIME_STEP)  # ✅ Vehicle moves after task execution
            self.env.add_remove_servers()  # ✅ Update RSUs dynamically


if __name__ == "__main__":
    realworld = RealWorld()
    realworld.run_simulation(num_tasks=10)  # ✅ Change: Simulates **10 tasks execution**
