import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  D = np.prod(x.shape[1:])
  x_reshaped = x.reshape(N, D)
  out = (w.T.dot(x_reshaped.T) + b.reshape(b.shape[0], 1)).T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  print(f"dout {dout.shape}")
  print(f"x {x.shape}")
  print(f"w {w.shape}")
  N = x.shape[0]
  input_shape = x.shape[1:]
  x_reshaped = x.reshape(N, np.prod(input_shape))

  dx = dout.dot(w.T).reshape(N, *input_shape)
  dw = dout.T.dot(x_reshaped).T#.reshape(N, *input_shape)
  db = np.sum(dout, axis=0)#.reshape(N, *input_shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[cache <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def convert(img):
  """
  input: img: (C, H, W)
  output: out: (H, W, C)
  """
  (C, H, W) = img.shape
  row_list = []
  for r in range(H):
    col_list = []
    for c in range(W):
      pixel = img[:, r, c]
      col_list.append(pixel)
    row_list.append(np.array(col_list))
  out = np.array(row_list)
  assert(out.shape == (H, W, C))
  return out

def pad_channel(pad, single_channel):
  return np.pad(single_channel, pad, "constant", constant_values=int(0))

def convolve(img, kernel, bias, stride, outDim):
  """
  img: (H, W, C)

  """
  (H, W, C) = img.shape
  (HH, WW, CC) = kernel.shape
  assert(C == CC)

  (outH, outW) = outDim
  out = np.zeros(outDim)
  # print(f"kernel: {kernel}")
  # print(f"H-HH = {H}-{HH} : {H-HH}")
  # print(f"stride: {stride}")
  for r in range(0, H-HH+1, stride):
    for c in range(0, W-WW+1, stride):
      out[int(r/stride)][int(c/stride)] = np.sum(img[r:r+WW,c:c+HH,:] * kernel) + bias
      # print(f"out[{r}, {c}]: {out[r][c]}")
  return out


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  (N, C, H, W) = x.shape
  (F, _, HH, WW) = w.shape
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param["stride"]
  pad = conv_param["pad"]
  print(f"stride: {stride}")
  print(f"pad: {pad}")
  outH = int(1 + (H + 2 * pad - HH) / stride)
  outW = int(1 + (W + 2 * pad - WW) / stride)

  #adjust kernel shapes
  new_w = np.zeros(shape=(F, HH, WW, C))
  for fi, f in enumerate(w):
    new_w[fi] = convert(f)

  #add padding to img and adjust img shape
  padded_img = np.zeros(shape=(N, H+pad*2, W+pad*2, C))
  for ii, img in enumerate(x):
    img_channels_first = []
    for chi, ch in enumerate(img):
      img_channels_first.append(pad_channel(pad, ch))
    padded_img[ii] = convert(np.array(img_channels_first))

  #convolve
  out = np.zeros(shape=(N, F, outH, outW))
  for ii, img in enumerate(padded_img):
    for ki, kernel in enumerate(new_w):
      out[ii][ki] = convolve(img, kernel, b[ki], stride, (outH, outW)) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  (N, C, H, W) = x.shape
  HH = pool_param["pool_height"]
  WW = pool_param["pool_width"]
  stride = pool_param["stride"]
  outW = int((W-WW)/stride+1)
  outH = int((H-HH)/stride+1)
  out = np.zeros((N, C, outH, outW))
  for ii, img in enumerate(x):
    for ci, ch in enumerate(img):
      for r in range(0, H-HH+1, stride):
        for c in range(0, W-WW+1, stride):
          out[ii][ci][int(r/stride)][int(c/stride)] = np.max(ch[r:r+HH, c:c+WW])
          #print(f"ch[{r}:{r+HH}][{c}:{c+WW}]: {ch[r:r+HH, c:c+WW]} --> {out[ii][ci][int(r/stride)][int(c/stride)]}")
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  (x, pool_param) = cache
  # print(f"dout shape: {dout.shape}")
  # print(f"x shape: {x.shape}")

  dx = np.zeros_like(x)

  (N, C, H, W) = x.shape
  HH = pool_param["pool_height"]
  WW = pool_param["pool_width"]
  stride = pool_param["stride"]
  for ii, img in enumerate(x):
    for ci, ch in enumerate(img):
      for r in range(0, H-HH+1, stride):
        for c in range(0, W-WW+1, stride):
          bigIndex = np.argmax(ch[r:r+HH, c:c+WW])
          bigR, bigC = int(bigIndex/stride), bigIndex%stride
          dx[ii][ci][r+bigR][c+bigC] = dout[ii][ci][int(r/stride)][int(c/stride)]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

