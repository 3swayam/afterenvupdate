import random
import numpy as np
from Config import Config


def genStates():
    """
    Generates dynamic state data for training and testing.

    State variables:
    - Task size (5M or 10M instructions)
    - Task deadline
    - Vehicle load factor (0-10)
    - RSU load factor (0-10)
    """
    states = []
    for task_size in [5, 10]:  # Change: Variable task size
        for deadline in range(3 if task_size == 5 else 5, 11):
            for vehicle_lf in range(0, 11):
                for rsu_lf in range(0, 11):
                    states.append([task_size, deadline, vehicle_lf, rsu_lf])

    # New: Shuffle states for randomness
    random.shuffle(states)

    # New: Split into training & testing sets (80-20 split)
    split_index = int(0.8 * len(states))
    train_states = states[:split_index]
    test_states = states[split_index:]

    print(f"Generated {len(train_states)} training states and {len(test_states)} testing states.")

    return train_states, test_states
