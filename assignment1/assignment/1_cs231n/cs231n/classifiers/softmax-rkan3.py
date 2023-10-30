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
  score = W.dot(X)
  score -= np.max(score) # for numeric stability
  exp_score = np.exp(score)
  #print(f"exp_scores are big?: {exp_score}")

  (k, N) = exp_score.shape
  score_sums = np.sum(exp_score, axis = 0) + 1e-32
  exp_score_norm = exp_score/score_sums

  idx = np.array(range(N))
  indices = (y, idx)
  loss = -np.sum(np.log(exp_score_norm[indices]))/N
  loss += 0.5 * reg * np.sum(W*W)

  exp_score_norm[indices] -= 1
  dW = (exp_score_norm.dot(X.T))/N + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW
