import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  s = W.dot(X)
  exp_s = np.exp(s)
  #print(f'W shape = {W.shape}')
  #print(f'X shape = {X.shape}')
  #print(f'exp_s shape = {exp_s.shape}')
  samples = exp_s.shape[1]
  k = exp_s.shape[0]
  norm = np.sum(exp_s, axis = 0)+1e-32
  exp_s_norm = exp_s/norm
  #print(f'exp_s_norm shape = {exp_s_norm.shape}')

  #print(f'norm shape = {norm.shape}')
  sf_max = np.zeros_like(y)
  idx = np.array(range(samples))
  indices = tuple([y,idx])
  exp_sj_norm = exp_s_norm[indices]
  #print(f'exp_sj_norm shape = {exp_sj_norm.shape}')
  sf_max = -np.log(exp_sj_norm)
  #print(f'sf_max shape = {sf_max.shape}')
  #print(f'samples = {samples}')
  #regularization should not include b, so b needs subtracted
  loss = np.sum(sf_max)/samples + 0.5*reg *(np.sum(np.power(W,2))-np.sum(np.power(W[:,0],2)))
  #loss = np.sum(sf_max)/samples + 0.5*reg *(np.sum(np.power(W,2)))
  exp_s_norm[indices] -= 1
  dW = (exp_s_norm.dot(X.T)/samples+reg*W)
  db = np.sum(exp_s_norm, axis = 1)/samples
  #update db
  dW[:,0] = db.T
  #print(f'db error = {error}')
  #print(f'dW shape = {dW.shape}')
  #print(f'dW[0] = {dW[0]}')
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
