
import numpy as np  # For numerical operations
from PIL import Image  # For image manipulation

from collections import deque
import copy

import torch  # PyTorch for tensor operations and DL
import torch.nn as nn
import torch.nn.functional as F  # Functional API of PyTorch for operations like softmax

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class DuelingDQN(nn.Module):
    def __init__(self, num_actions, input_shape=(4, 84, 84), dropout=0.2):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        
        # Convolutional layers from SharedFeatureNetwork
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(dropout)
        
        # Dynamically calculate the size of the flattened features after the conv layers
        self._feature_size = self._get_conv_output(input_shape)
        
        # Fully connected layers for Dueling DQN
        self.features_fc = nn.Linear(self._feature_size, 512)
        self.value_stream = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_actions))
        
    
    def forward(self, x):
        
        # Apply conv layers and dropout as in SharedFeatureNetwork
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        # Apply fully connected layers as in DuelingDQN
        x = torch.relu(self.features_fc(x))
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine value and advantages to get Q values
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = F.relu(self.conv1(input))
            output = F.relu(self.conv2(output))
            output = F.relu(self.conv3(output))
            output = output.view(output.size(0), -1)
            return output.size(1)
        

class Agent(object):
    def __init__(self):
        # Define the device
        self.device = torch.device("cpu")
        self.action_size = len(COMPLEX_MOVEMENT)  # Make sure COMPLEX_MOVEMENT is defined somewhere
        self.num_stack = 2
        self.state_shape = (self.num_stack, 84, 84)  # Adjusted to match the SharedFeatureNetwork input_shape
        
        # Initialize the main and target networks with the updated Dueling DQN architecture
        self.main_DQN = DuelingDQN(num_actions=12, input_shape=self.state_shape).to(self.device)
        self.main_DQN.eval()
            
        self.checkpoint = './111003804_hw2_data.py'
        self.load_checkpoint(self.checkpoint)
        self.temperature = 10

        self.play_states = deque(maxlen=4)
        self.skip = 4
        
        self.last_action = np.random.choice(self.action_size)
        self.play_step = 0
    
    
    def _preprocess_state(self, states):
        """Process a list of states into a self.num_stackx84x84 normalized grayscale tensor.
        If fewer than self.num_stack states are provided, repeat the first state by inserting it at the beginning
        until there are self.num_stack states."""

        states = states[-self.num_stack:]
        # Ensure states list has self.num_stack elements, repeating the first element if necessary
        while len(states) < self.num_stack:
            states.insert(0, states[0])  # Insert the first state at the beginning

        processed_states = []

        for state in states:
            # Convert to PIL Image for resizing and grayscale conversion
            state_image = Image.fromarray(state).convert('L')  # 'L' mode for grayscale

            # Resize
            state_resized = state_image.resize((84, 84), Image.Resampling.LANCZOS)

            # Convert back to NumPy array and normalize
            state_np = np.array(state_resized, dtype=np.float32) / 255.0  # Normalized to [0, 1]

            processed_states.append(state_np)

        # Stack processed states to form a single tensor with shape (self.num_stack, 84, 84)
        state_tensor = np.stack(processed_states, axis=0)

        return state_tensor.astype(np.float32)

    
    
    def act(self, state):
        self.play_states.append(copy.deepcopy(state))
        if np.random.rand() <= 0.05:
            # Exploration: choose a random action
            return np.random.choice(self.action_size)

        preprocessed_states = self._preprocess_state(list(self.play_states))
        # Convert the state to a PyTorch tensor and add a batch dimension
        state_tensor = torch.FloatTensor(preprocessed_states).unsqueeze(0).to(self.device)

        # Forward pass to get Q-values
        q_values = self.main_DQN(state_tensor)
        action = q_values.argmax().item()  # Get the action with the highest Q-value

        return action

    # Load model and optimizer state
    def load_checkpoint(self, filename):
        # Load the saved checkpoint
        checkpoint = torch.load(filename, map_location=self.device)
        self.main_DQN.load_state_dict(checkpoint['main_DQN_state_dict'])

