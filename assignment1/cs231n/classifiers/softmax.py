import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    
    scores = np.exp(scores)
    loss += -np.log(scores[y[i]]/np.sum(scores))
    for j in range(num_class):
        if j == y[i]:
            dW[:, j] += - X[i] + 1/np.sum(scores)*scores[y[i]]*X[i]
        else:
            dW[:, j] += 1/np.sum(scores)*scores[j]*X[i]
 
  loss /= num_train
  dW /= num_train
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)

  scores = np.exp(scores)
  indexs = [i for i in range(num_train)]

  loss += np.sum(-np.log(scores[indexs, y] / np.sum(scores, axis=1)))
    
  grad_scores = (1/np.sum(scores, axis=1).reshape(num_train, -1)) * scores
  grad_scores[indexs, y] = grad_scores[indexs, y] - 1
  dW += np.dot(X.T, grad_scores)
  
  loss /= num_train
  dW /= num_train
    
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

