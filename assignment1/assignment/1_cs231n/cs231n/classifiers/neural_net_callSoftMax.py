import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.softmax import *

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape
  samples = N
  input_features = D
  #print(f'num samples = {samples}')
  #print(f'input features = {input_features}')
  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  #print(f'W1 shape = {W1.shape}')
  #print(f'b1 shape = {b1.shape}')
  #print(f'X shape = {X.shape}')
  H1 = W1.shape[1]
  b1t = b1.T
  b1t = b1t.reshape(1,H1)
  W1b1 = np.concatenate((b1t,W1),axis=0)
  H2 = W2.shape[1]
  b2t = b2.T
  b2t = b2t.reshape(1,H2)
  W2b2 = np.concatenate((b2t,W2),axis=0)
  Xt = X.T
  ones = np.ones((1,N))
  #print(f'ones shape = {ones.shape}')
  Xt1 = np.concatenate((ones,Xt),axis=0)
  Xextend = Xt1.T
  #print(f'W1b1 shape = {W1b1.shape}')
  print(f'W2 shape = {W2.shape}')
  print(f'W2b2 shape = {W2b2.shape}')
  #print(f'Xextend shape = {Xextend.shape}')
  z = np.dot(Xextend,W1b1)
  z_act = np.maximum(np.zeros(z.shape),z)
  #print(f'z_act shape = {z_act.shape}')
  z_act_ext = np.concatenate((ones,z_act.T),axis=0).T
  #print(f'z_act_ext shape = {z_act_ext.shape}')
  scores = np.dot(z_act_ext,W2b2)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss, grad = softmax_loss_vectorized(W2b2.T, z_act_ext.T, y, reg)
  #print(f'loss = {loss}')
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  loss += 0.5*reg *(np.sum(np.power(W1,2)))
  #print(f'loss = {loss}')
  grads = {}
  grads['W2'] = grad.T[1:,:]
  grads['b2'] = grad.T[0,:]
  print(f'grads[W2] shape = {grad.T[1:,:].shape}')
  print(f'grads[b2] shape = {grad.T[0,:].shape}')
  print(f'W1 shape = {W1.shape}')
  print(f'b1 shape = {b1.shape}')
  print(f'W1b1 shape = {W1b1.shape}')
  grads['W1'] = np.dot(W2,grad.T[1:,:].T)
  grads['b1'] = np.dot(W2,grad.T[0,:])
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  pass
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads

