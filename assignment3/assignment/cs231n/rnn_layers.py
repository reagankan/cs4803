from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    N, D = x.shape
    D, H = Wx.shape
    next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b.reshape(1, H))
    cache = (next_h, x, prev_h, Wx, Wh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache, verbose=True):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################

    next_h, x, prev_h, Wx, Wh = cache
    N, D = x.shape
    D, H = Wx.shape
    ds = {N:"N", D:"D", H:"H"}
    def sh(s, d=ds):
        return f"({d[s[0]]}, {d[s[1]]})"
    dTanh = 1 - np.power(next_h, 2)
    dSum = np.multiply(dnext_h, dTanh) #element-wise
    dx = dSum.dot(Wx.T)
    dWx = x.T.dot(dSum)

    dprev_h = dSum.dot(Wh) #bad
    dWh = dSum.T.dot(prev_h) #bad
    dprev_h = dSum.dot(Wh.T)
    dWh = prev_h.T.dot(dSum)

    db = np.ones((1, N)).dot(dSum).T.reshape(H,)
    #db = np.sum(dSum, axis=0).T.reshape(H,)
    if verbose:
        print("N %d, D %d, H %d" % (N, D, H))
        print(f"dnext_h: {sh(dnext_h.shape)} {dnext_h.shape}")
        print(f"x: {sh(x.shape)} {x.shape}")
        print(f"prev_h: {sh(prev_h.shape)} {prev_h.shape}")
        print(f"Wx: {sh(Wx.shape)} {Wx.shape}")
        print(f"Wh: {sh(Wh.shape)} {Wh.shape}")
        print(f"dTanH: {sh(dTanh.shape)} {dTanh.shape}")
        print(f"dSum: {sh(dSum.shape)} {dSum.shape}")
        print(f"dx: {dx.shape}")
        print(f"dprev_h: {dprev_h.shape}")
        print(f"dWx: {dWx.shape}")
        print(f"dWh: {dWh.shape}")
        print(f"db: {db.shape}")
    #dx = 1 - np.power(dx, 2)
    #dprev_h = 1 - np.power(dprev_h, 2)
    #dWx = 1 - np.power(dWx, 2)
    #dWh = 1 - np.power(dWh, 2)
    #db = 1 - np.power(db, 2)
    #assert( np.array_equal(dx, ))
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    _, H = h0.shape
    prev_h = h0
    h, cache = [], []
    for t in range(T):
        prev_h, cache_t = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h.append(prev_h)
        cache.append(cache_t)
        #print("x[t] type: %s" % type(x[:,t,:]))
        #print("prev_h type: %s" % type(prev_h))
        #print(f"prev_h shape: {prev_h.shape}")
        #print("Wx type: %s" % type(Wx))
        #print("Wh type: %s" % type(Wh))
        #print("b type: %s" % type(b))
    h = np.array(h).swapaxes(0, 1) #reshape(N, T, H) is BAD.
    #print(f"h: {h}")
    cache = np.array(cache)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
    - cache: ???

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    D, _ = cache[0][3].shape
    dx = []
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    dprev_h_t = 0
    for t in range(T-1, -1, -1):
        dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dh[:, t, :] + dprev_h_t, cache[t], verbose=False)
        dx = [dx_t] + dx
        dh0 = dprev_h_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dx = np.array(dx).swapaxes(0, 1)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out = W[x[:, :]]
    cache = (W, x)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    _, _, D = dout.shape
    W, x = cache
    V, _ = W.shape
    dW = np.zeros((V, D))
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape
    a = x.dot(Wx) + prev_h.dot(Wh) + b #(N, 4H)
    a_i = a[:, :H]
    a_f = a[:, H:2*H]
    a_o = a[:, 2*H:3*H]
    a_g = a[:, 3*H:]
    i = sigmoid(a_i)
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g)

    next_c = np.multiply(f, prev_c) + np.multiply(i, g)
    tanh_next_c = np.tanh(next_c)
    next_h = np.multiply(o, tanh_next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, b, a, a_i, a_f, a_o, a_g, i, f, o, g, tanh_next_c, next_c, next_h) #keep everything just in case
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache, verbose=False):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    N, H = dnext_h.shape
    x, prev_h, prev_c, Wx, Wh, b, a, a_i, a_f, a_o, a_g, i, f, o, g, tanh_next_c, next_c, next_h = cache

    dL_do = np.multiply(dnext_h, tanh_next_c)

    dNextH_dTanhNextC = np.multiply(o, 1-np.power(tanh_next_c, 2))
    dL_dTanhNextC = np.multiply(dnext_h, dNextH_dTanhNextC)

    dL_dNextC = dnext_c + dL_dTanhNextC
    dc = np.multiply(dL_dNextC, f)

    dL_di = np.multiply(dL_dNextC, g)
    dL_df = np.multiply(dL_dNextC, prev_c)
    dL_dg = np.multiply(dL_dNextC, i)

    #now with dL_d(ifog), need to get past activation blocks(pab)
    dL_di_pab = np.multiply(dL_di, np.multiply(i, 1 - i))
    dL_df_pab = np.multiply(dL_df, np.multiply(f, 1 - f))
    dL_do_pab = np.multiply(dL_do, np.multiply(o, 1 - o)) #sigmoid
    dL_dg_pab = np.multiply(dL_dg, 1 - np.power(g, 2)) #tanh
    #dSum: upstream grad for x.Wx + h.Wh + b
    dSum = np.concatenate((dL_di_pab, dL_df_pab, dL_do_pab, dL_dg_pab), axis=1)
    if verbose:
        print(f"dL/di {dL_di_pab.shape}")
        print(f"dSum {dSum.shape}")
        print(f"sum, aka a {a.shape}")

    # the rest is the same as vanilla rnn
    dx = dSum.dot(Wx.T)
    dWx = x.T.dot(dSum)
    dh = dSum.dot(Wh.T)
    dWh = prev_h.T.dot(dSum)
    db = np.ones((1, N)).dot(dSum).T.reshape(4*H,)

    dprev_h = dh
    dprev_c = dc
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # same as rnn, just added prev_c variable.
    N, T, D = x.shape
    _, H = h0.shape
    prev_h = h0
    prev_c = np.zeros_like(h0)
    h, cache = [], []
    for t in range(T):
        prev_h, prev_c, cache_t = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        h.append(prev_h)
        cache.append(cache_t)
    h = np.array(h).swapaxes(0, 1) #reshape(N, T, H) is BAD.
    cache = np.array(cache)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    _, D = cache[0][0].shape
    dx = []
    dh0 = np.zeros((N, 4*H)) #just 0 would've worked too.
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    dprev_h_t = 0
    dprev_c_t = 0
    for t in range(T-1, -1, -1):
        dx_t, dprev_h_t, dprev_c_t, dWx_t, dWh_t, db_t = lstm_step_backward(dh[:, t, :] + dprev_h_t, dprev_c_t, cache[t], verbose=False)
        dx = [dx_t] + dx
        dh0 = dprev_h_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dx = np.array(dx).swapaxes(0, 1)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
