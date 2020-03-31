from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *
from cs231n.classifiers.rnn import CaptioningRNN

class MyModel(CaptioningRNN):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, caption_dim=17, cell_type='rnn', dtype=np.float32):
        super().__init__(word_to_idx, input_dim, wordvec_dim, hidden_dim, cell_type, dtype)

        self.hidden_dim = hidden_dim #H
        self.wordvec_dim = wordvec_dim
        self.caption_dim = caption_dim - 1 #T
        #print(f"hidden_dim: {self.hidden_dim}")

        #TODO: add pre_lstm and post_lstm affine layers W and b.
        self.V, self.W = self.params["W_embed"].shape
        # print(f"W: {self.W}")
        # print(f"V: {self.W}")
        # print(f"wordvec_dim: {self.wordvec_dim}")
        dim = self.caption_dim * self.W
        self.params['W_prelstm'] = np.random.randn(dim , dim)
        self.params['W_prelstm'] /= np.sqrt(dim)
        self.params['b_prelstm'] = np.zeros(dim)

        dim = self.caption_dim * self.hidden_dim
        self.params['W_postlstm'] = np.random.randn(dim, dim)
        self.params['W_postlstm'] /= np.sqrt(dim)
        self.params['b_postlstm'] = np.zeros(dim)

    def loss(self, features, captions, verbose=False):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        # print(f"captions: {captions.shape}")
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        # shapes
        N, D = features.shape
        _, T = captions.shape
        _, H = W_proj.shape
        V, W = W_embed.shape

        #forward pass, 5 steps.
        #**1. image features (N, D) -> hidden state (N, H)
        h0, af_cache = affine_forward(features, W_proj, b_proj)

        #**2. captions_in --wordembedded--> vector (N, T, W)
        embeded, we_cache = word_embedding_forward(captions_in, W_embed)
        # print(f"embeded: {embeded.shape}")
        # print(f"hidden: {(self.wordvec_dim, self.wordvec_dim)}")

        ##TODO: add feedforward layer here
        # don't forget relu activations
        W_prelstm, b_prelstm = self.params['W_prelstm'], self.params['b_prelstm']
        prelstm, prelstm_cache = affine_forward(embeded, W_prelstm, b_prelstm, False)

        new_embeded, relu1_cache = relu_forward(prelstm)
        new_embeded = new_embeded.reshape(embeded.shape)


        #**3. hidden state vectors (N, T, H)
        if self.cell_type == "rnn":
            hidden_states, forward_cache = rnn_forward(new_embeded, h0, Wx, Wh, b)
        elif self.cell_type == "lstm":
            hidden_states, forward_cache = lstm_forward(embeded, h0, Wx, Wh, b)

        # print(f"hidden states: {hidden_states.shape}")
        ##TODO: add feedforward layer here
        W_postlstm, b_postlstm = self.params['W_postlstm'], self.params['b_postlstm']
        postlstm, postlstm_cache = affine_forward(hidden_states, W_postlstm, b_postlstm, False)

        new_hidden_states, relu2_cache = relu_forward(postlstm)
        new_hidden_states = new_hidden_states.reshape(hidden_states.shape)

        #**4. compute scores (N, T, V)
        scores, scores_cache = temporal_affine_forward(new_hidden_states, W_vocab, b_vocab)
        # print(f"scores: {scores.shape}")

        #**5. compute loss from captions_out(as gt)
        loss, dx = temporal_softmax_loss(scores, captions_out, mask)

        #TODO: backward
        grads["W_prelstm"] = np.zeros(self.params["W_prelstm"].shape)
        grads["b_prelstm"] = np.zeros(self.params["b_prelstm"].shape)
        grads["W_postlstm"] = np.zeros(self.params["W_postlstm"].shape)
        grads["b_postlstm"] = np.zeros(self.params["b_postlstm"].shape)
        #print(self.params.keys())
        #dict_keys(['W_embed', 'W_proj', 'b_proj', 'Wx', 'Wh', 'b', 'W_vocab', 'b_vocab'])
        up_grad = dx
        up_grad, dw, db = temporal_affine_backward(up_grad, scores_cache) #args: upstream_grad, and cache
        grads["W_vocab"] = dw
        grads["b_vocab"] = db

        #TODO: add postlstm gradients
        # print(f"up_grad: {up_grad.shape}")
        # print(f"relu2_cache: {relu2_cache.shape}")
        up_grad = relu_backward(up_grad.reshape(relu2_cache.shape[0], -1), relu2_cache)
        up_grad, dw, db = affine_backward(up_grad, postlstm_cache)
        grads["W_postlstm"] = dw
        grads["b_postlstm"] = db
        #x.reshape(x.shape[0], -1)

        if self.cell_type == "rnn":
            up_grad, dh0, dWx, dWh, db = rnn_backward(up_grad, forward_cache)
        elif self.cell_type == "lstm":
            up_grad, dh0, dWx, dWh, db = lstm_backward(up_grad, forward_cache)
        grads["Wx"] = dWx
        grads["Wh"] = dWh
        grads["b"] = db

        #TODO: add prelstm gradients
        up_grad = relu_backward(up_grad.reshape(relu1_cache.shape[0], -1), relu1_cache)
        up_grad, dw, db = affine_backward(up_grad, prelstm_cache)
        grads["W_prelstm"] = dw
        grads["b_prelstm"] = db

        #ordering of next two aren't important
        dw = word_embedding_backward(up_grad, we_cache)
        grads["W_embed"] = dw

        up_grad = dh0
        up_grad, dw, db = affine_backward(up_grad, af_cache)
        grads["W_proj"] = dw
        grads["b_proj"] = db

        if verbose:
            print(f"N: {N}, D: {D}, T: {T}, H: {H}, V: {V}, W: {W}")
            def sh(s, d={N:"N", D:"D", T:"T",T-1:"T-1", H:"H", V:"V", W:"W"}):
                mid = ", ".join([str(d[dim]) for i, dim in enumerate(s)])
                return "(" + mid + ")"
            print(f"image features {sh(features.shape)} -> hidden state {sh(h0.shape)}")
            print(f"embeded shape: {embeded.shape}")
            print(f"captions_in shape: {captions_in.shape}")
            print(f"captions_in ----> vector {sh(embeded.shape)}")
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads