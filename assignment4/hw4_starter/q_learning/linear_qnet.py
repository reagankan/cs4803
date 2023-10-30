import numpy as np

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, env, config, debug=False):
        """
        A state-action (Q) network with a single fully connected
        layer, takes the state as input and gives action values
        for all actions.
        """
        super().__init__()

        #####################################################################
        # TODO: Define a fully connected layer for the forward pass. Some
        # useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        self.H, self.W, self.C = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.o_per_s = config.state_history

        
        self.in_features = self.H * self.W * self.C * self.o_per_s
        self.out_features = self.num_actions
        self.layer = nn.Linear(self.in_features, self.out_features)
        if debug:
            print(f"num actions: {self.num_actions}")
            print(f"H: {self.H}")
            print(f"W: {self.W}")
            print(f"C: {self.C}")
            print(f"o_per_s: {self.o_per_s}")
            print(self.layer)
            print(self.layer.weight)
            print(self.layer.weight.shape)
            print(type(self.layer.weight))
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state, debug=False):
        """
        Returns Q values for all actions

        Args:
            state: tensor of shape (batch, H, W, C x config.state_history)

        Returns:
            q_values: tensor of shape (batch_size, num_actions)
        """
        #####################################################################
        # TODO: Implement the forward pass, 1-2 lines.
        #####################################################################
        batch_size, _, _, _ = state.shape
        state = state.view(batch_size, -1)
        out = self.layer(state)
        return out
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
