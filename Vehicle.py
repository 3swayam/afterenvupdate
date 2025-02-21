from Config import Config
import math
import numpy as np
from collections import deque


class Vehicle:
    def __init__(self):
        self.freq = Config.VEHICLE_FREQUENCY
        self.power = Config.VEHICLE_POWER
        self.loadfactor = 0
        self.CPI = Config.VEHICLE_CPI

        self.x_position, self.y_position = Config.VEHICLE_INITIAL_POSITION
        self.velocity_x, self.velocity_y = Config.VEHICLE_INITIAL_VELOCITY

        self.position_queue = deque(maxlen=Config.QUEUE_SIZE)
        self.path_points = Config.VEHICLE_PATH_POINTS
        self.current_waypoint = 1

        # ✅ New: Track total energy consumed
        self.total_energy_consumed = 0

    def compDelay(self, task_size):
        return task_size / (self.freq / self.CPI)

    def compute_energy(self, task_size, comm_delay=0):
        comp_energy = self.power * self.compDelay(task_size)
        comm_energy = self.power * comm_delay

        # ✅ Track energy consumption
        total_energy = comp_energy + comm_energy
        self.total_energy_consumed += total_energy

        return total_energy

    def get_energy_consumed(self):  # ✅ New method
        """ Returns total energy consumed by the vehicle """
        return self.total_energy_consumed

    def get_state(self):
        task_size = 5
        task_deadline = 10
        rsu_load_factor = 5

        return [task_size, task_deadline, self.loadfactor, rsu_load_factor]

    def move(self, time_step):
        if self.current_waypoint < len(self.path_points):
            target_x, target_y = self.path_points[self.current_waypoint]
            direction_x = target_x - self.x_position
            direction_y = target_y - self.y_position
            distance = math.sqrt(direction_x ** 2 + direction_y ** 2)

            if distance > Config.VEHICLE_PRECISION_ERROR:
                unit_x = direction_x / distance
                unit_y = direction_y / distance
                self.x_position += unit_x * self.velocity_x * time_step
                self.y_position += unit_y * self.velocity_y * time_step

                if distance < 100:
                    self.velocity_x = max(self.velocity_x - Config.VEHICLE_DECELERATION * time_step, 5)
                else:
                    self.velocity_x = min(self.velocity_x + Config.VEHICLE_ACCELERATION * time_step,
                                          Config.VEHICLE_SPEED / 3.6)

            if distance < Config.VEHICLE_PRECISION_ERROR:
                self.current_waypoint += 1

        self.position_queue.append((self.x_position, self.y_position))

    def stayTime(self, rsu):
        return rsu.radius / max(self.velocity_x, 1e-6)

    def isConnected(self, rsu):
        return math.sqrt((self.x_position - rsu.x_position) ** 2 + (self.y_position - rsu.y_position) ** 2) < rsu.radius

    def computeInstructionCycles(self, cycles_per_instruction=Config.VEHICLE_CPI):
        return (self.freq * cycles_per_instruction) / 200

    def resourceUtilization(self):
        matrix = np.random.rand(*Config.COMPUTATION_MATRIX_SIZE)
        utilization = np.sum(matrix) / np.prod(Config.COMPUTATION_MATRIX_SIZE)
        return utilization <= Config.RESOURCE_UTILIZATION_THRESHOLD
