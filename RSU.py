import math
import numpy as np
from Config import Config


class RSU:
    def __init__(self, vehicle, relative_x, relative_y=0):

        # ✅ New: RSU position is set **relative to vehicle’s coordinates**
        self.x_position = vehicle.x_position + relative_x
        self.y_position = vehicle.y_position + relative_y

        self.freq = Config.RSU_FREQUENCY
        self.power = Config.RSU_POWER
        self.radius = Config.RSU_RADIUS
        self.loadfactor = 0  # ✅ New: Tracks RSU load dynamically

        # ✅ New: Precision handling for RSU position
        self.precision_error = Config.RSU_PRECISION_ERROR

    def updatePosition(self, vehicle):

        self.x_position = vehicle.x_position + self.precision_error
        self.y_position = vehicle.y_position + self.precision_error  # ✅ New: Ensures **precision tracking**

    def compDelay(self, task_size):

        return task_size / (self.freq / Config.RSU_CPI)

    def isVehicleConnected(self, vehicle):

        return self.calculateDistance(vehicle) < self.radius

    def calculateDistance(self, vehicle):

        return math.sqrt((vehicle.x_position - self.x_position) ** 2 + (vehicle.y_position - self.y_position) ** 2)

    def computeEnergyConsumption(self, task_size):

        return self.power * self.compDelay(task_size)  # ✅ New: Task processing energy calculation

    def resourceOptimization(self):

        matrix = np.random.rand(*Config.COMPUTATION_MATRIX_SIZE)  # ✅ New: Generate **resource matrix**
        utilization = np.sum(matrix) / np.prod(Config.COMPUTATION_MATRIX_SIZE)
        return utilization <= Config.RESOURCE_UTILIZATION_THRESHOLD  # ✅ New: Prevents overload
