from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


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

        #**3. hidden state vectors (N, T, H)
        if self.cell_type == "rnn":
            hidden_states, forward_cache = rnn_forward(embeded, h0, Wx, Wh, b)
        elif self.cell_type == "lstm":
            hidden_states, forward_cache = lstm_forward(embeded, h0, Wx, Wh, b)

        #**4. compute scores (N, T, V)
        scores, scores_cache = temporal_affine_forward(hidden_states, W_vocab, b_vocab)

        #**5. compute loss from captions_out(as gt)
        loss, dx = temporal_softmax_loss(scores, captions_out, mask)

        #backward
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #

        #print(self.params.keys())
        #dict_keys(['W_embed', 'W_proj', 'b_proj', 'Wx', 'Wh', 'b', 'W_vocab', 'b_vocab'])
        up_grad = dx
        up_grad, dw, db = temporal_affine_backward(up_grad, scores_cache) #args: upstream_grad, and cache
        grads["W_vocab"] = dw
        grads["b_vocab"] = db

        if self.cell_type == "rnn":
            up_grad, dh0, dWx, dWh, db = rnn_backward(up_grad, forward_cache)
        elif self.cell_type == "lstm":
            up_grad, dh0, dWx, dWh, db = lstm_backward(up_grad, forward_cache)
        grads["Wx"] = dWx
        grads["Wh"] = dWh
        grads["b"] = db

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


    def sample(self, features, max_length=30, verbose=False):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        N, D = features.shape
        T = max_length

        prev_c = 0 #for lstm
        prev_h, af_cache = affine_forward(features, W_proj, b_proj)
        prev_word = np.zeros((N, 1)) + self._start
        prev_word = prev_word.astype(int)
        if verbose:
            print(f"captions: {captions.shape}")
            print(f"prev_h : {prev_h.shape}")
            print(f"W_vocab: {W_vocab.shape}")
            print(f"prev_word: {prev_word}")
        for t in range(T):
            embeded_word, we_cache = word_embedding_forward(prev_word, W_embed)
            if verbose:
                print(f"iter: {t}")
                print(f"embeded word: {embeded_word.shape}")
                print(f"prev_h : {prev_h.shape}")
            if self.cell_type == "rnn":
                prev_h, cache_t = rnn_step_forward(embeded_word[:,0,:], prev_h, Wx, Wh, b) #T = 1
            elif self.cell_type == "lstm":
                prev_h, prev_c, cache_t = lstm_step_forward(embeded_word[:,0,:], prev_h, prev_c, Wx, Wh, b) #T = 1

            d1, d2 = prev_h.shape
            prev_h_pad = prev_h.reshape(d1, 1, d2)
            scores, scores_cache = temporal_affine_forward(prev_h_pad, W_vocab, b_vocab)
            # print(f"vocab size: {len(self.word_to_idx)}")
            # print(f"scores: {scores.shape}")
            # print(f"W_vocab: {W_vocab.shape}")
            ind = np.argmax(scores, axis=2)
            sampled_word = W_vocab[:,ind]
            if verbose:
                print(f"prev_h : {prev_h.shape}")
                print(f"prev_h_pad : {prev_h_pad.shape}")
                print(f"scores: {scores.shape}")
                print(f"ind: {ind.shape}")
                print(f"sampled_word: {sampled_word.shape}")
                print(f"captioned_wd: {captions.shape}")
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
