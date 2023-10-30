import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        # 

        # We now describe the exact architecture used for all seven Atari games. The input to the neural
        # network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
        # filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
        # hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
        # final hidden layer is fully-connected and consists of 256 rectifier units.
        # The output layer is a fullyconnected linear layer with a single output for each valid action. The number of valid actions varied
        # between 4 and 18 on the games we considered. We refer to convolutional networks trained with our
        # approach as Deep Q-Networks (DQN).
        #####################################################################
        def output_dim(H1, W1, kernel_size, padding, stride):
            F = kernel_size
            P = padding
            S = stride
            W2=(W1-F+2*P)//S+1
            H2=(H1-F+2*P)//S+1
            return H2, W2

        H, W, C = env.observation_space.shape
        in_channels = C*config.state_history
        hidden_dim = 16
        kernel_size = 8
        stride = 4
        padding = 0#(kernel_size - 1) // 2
        # print(f"conv1 shape: {in_channels}.")
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)

        conv1H, conv1W = output_dim(H, W, kernel_size, padding, stride)
        # print(f"conv1 output: {conv1H}, {conv1W}")

        in_channels = hidden_dim
        hidden_dim = 32
        kernel_size = 4
        stride = 2
        padding = 0#(kernel_size - 1) // 2
        self.conv2 = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        conv2H, conv2W = output_dim(conv1H, conv1W, kernel_size, padding, stride)
        # print(f"conv2 output: {conv2H}, {conv2W}")
         
        #inf = int(W1*W2*hidden_dim)
        #print(f"input features: {inf}")
        outf = 256
        # print(f"input final hidden: {conv2H * conv2W * hidden_dim}")
        self.final_hidden = nn.Linear(conv2H * conv2W * hidden_dim, outf)

        self.num_actions = env.action_space.n
        inf = outf
        outf = env.action_space.n #num actions
        self.output_layer = nn.Linear(inf, outf)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        # print(f"state.shape: {state.shape}")
        N, H, W, C = state.shape
        state = state.view(N, C, H, W)
        out = self.conv1(state)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        # print(f"conv2 output.shape: {out.shape}")
        N, C, H, W = out.shape
        out = out.view(N, -1)
        out = self.final_hidden(out)
        out = self.output_layer(out)

        # print(f"final output shape: {out.shape}")
        # print(f"num_actions: {self.num_actions}")
        return out
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
