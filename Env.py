from Config import Config
from RSU import RSU
import random
from reward import reward_calculate

class Env:
    def __init__(self, vehicle,states,train_mode=True):

        self.vehicle = vehicle
        self.rsus = []  # ✅ Change: List to store multiple RSUs dynamically
        self.initializeServers()  # ✅ New: Setup initial RSUs
        self.cnt = 0
        self.states = states
        self.action_space = [0,1] # 0-local, 1-offload
        self.train_mode = train_mode
    
    def reset(self):
        self.delay = 0
        self.vehicle.loadfactor = 0
        self.loadfactor = 0
        self.cnt = 0
        return self.states[0]

    def initializeServers(self):

        for i in range(3):  # ✅ Start with at least **3 RSUs**
            self.rsus.append(RSU(self.vehicle, relative_x=(i + 1) * Config.RSU_MIN_DISTANCE))

    def add_remove_servers(self):

        # ✅ Remove RSUs that are out of range
        self.rsus = [rsu for rsu in self.rsus if rsu.isVehicleConnected(self.vehicle)]

        # ✅ Ensure at least one RSU is available
        if not self.rsus or len(self.rsus) < 1:  # ✅ Keep at least **two active RSUs**
            new_rsu_x = self.vehicle.x_position + random.randint(Config.RSU_MIN_DISTANCE, Config.MAX_RSU_RANGE)
            self.rsus.append(RSU(self.vehicle, relative_x=new_rsu_x))  # ✅ New RSU ahead of vehicle

    def selectClosestServer(self):

        return min(self.rsus, key=lambda rsu: rsu.calculateDistance(self.vehicle))
    
    
    def step(self, state,action):
        self.vehicle.loadfactor = state[2]
        self.done = False
        closest_RSU = self.selectClosestServer()
        closest_RSU.loadfactor = state[3]
        
        reward,cost,latency = reward_calculate(action, state,self.vehicle,
                                       closest_RSU)

        # ✅ Move vehicle & update servers
        self.vehicle.move(Config.TIME_STEP)
        self.add_remove_servers()

        # ✅ Generate next state
        self.cnt+=1
        self.next_state = self.states[self.cnt]

        if self.train_mode: 
            if(self.cnt == Config.N_TRAIN_STEPS_PER_EPISODE):
                self.done = True
        else:
           if(self.cnt == Config.N_TEST_STEPS_PER_EPISODE):
                self.done = True

        return self.next_state, reward, self.done,cost,latency


