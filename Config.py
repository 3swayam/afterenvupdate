class Config(object):
    MIN_NUMBER_RSU = 3
    TOTAL_AVAIL_RSU = 10
    # 🚗 VEHICLE PARAMETERS
    VEHICLE_FREQUENCY = 300 * 1e6  # Hz (300 MHz)
    VEHICLE_SPEED = 60  # km/h
    VEHICLE_POWER = 10  # Watts
    VEHICLE_CPI = 2  # Cycles per instruction

    VEHICLE_ACCELERATION = 2  # New: Acceleration for velocity tracking
    VEHICLE_DECELERATION = 2  # New: Deceleration to simulate slowing down
    VEHICLE_INITIAL_POSITION = (0, 0)  # Change: Origin changed to (0,0)
    VEHICLE_INITIAL_VELOCITY = (16.67, 0)  # Change: Velocity in x-direction, stationary in y

    VEHICLE_PRECISION_ERROR = 1e-6  # New: Precision handling for position calculations
    VEHICLE_PATH_POINTS = [(0, 0), (500, 0), (1000, 0)]  # New: Path stored as waypoints
    MAX_DISTANCE = 1000

    # 📡 RSU PARAMETERS
    RSU_FREQUENCY = 2 * 1e9  # Hz (2 GHz)
    RSU_HEIGHT = 7  # meters
    RSU_RADIUS = 600  # meters
    RSU_POWER = 15  # Watts
    RSU_CPI = 2  # Cycles per instruction
    BANDWIDTH = 50 * 1e6  # Hz (50 MHz)

    MAX_RSU_RANGE = 500  # Change: Increased RSU range for better connectivity
    RSU_MIN_DISTANCE = 250  # New: Minimum distance before adding a new RSU
    RSU_PRECISION_ERROR = 1e-6  # New: Precision handling for RSU position calculations


    # NEW 
    PATH_LOSS_EXPONENT = -3
    RAYLEIGH_FADING_CHANNEL = 0.5
    NOISE = -90 # in dB
    NOISE = 10**(NOISE/10)
    LATENCY = 5*1e-3

    # 🎯 REINFORCEMENT LEARNING PARAMETERS
    STATE_SIZE = 4
    ACTION_SIZE = 2
    DISCOUNT = 0.99
    EPS_MAX = 1.0
    EPS_MIN = 0.01
    EPS_DECAY = 0.995
    MEMORY_CAPACITY = 5000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_TRAIN_EPS = 1000
    NUM_TEST_EPS = 200
    TRAIN_MODE = True
    MODEL_NAME = "optimized_multi_agent_model"

    NUM_MEMORY_FILL_EPS = 3
    N_TRAIN_STEPS_PER_EPISODE = 1350
    N_TEST_STEPS_PER_EPISODE = 330

    # ⏳ REAL-TIME TRAINING & EXECUTION
    TIME_STEP = 1  # New: 1 second per simulation step
    TASK_EXECUTION_INTERVAL = 150  # New: Execute tasks every 150s
    MAX_EXECUTION_TIME = 100_000_000  # New: Up to 100 million seconds
    EXECUTION_ITERATIONS = 10_000  # New: Iterations for accuracy
    UPDATE_FREQUENCY = 100  # Defines how often the target network is updated

    # 📊 COMPUTATION & RESOURCE MANAGEMENT
    COMPUTATION_MATRIX_SIZE = (5, 5)  # New: Matrix for resource optimization
    VEHICLE_COST_FACTOR = 0.01  # New: Local processing cost factor
    RSU_COST_FACTOR = 0.005  # New: RSU offloading cost factor
    RESOURCE_UTILIZATION_THRESHOLD = 0.8  # New: Buffer threshold for resource allocation
    QUEUE_SIZE = 100  # New: Queue-based system for real-time updates
