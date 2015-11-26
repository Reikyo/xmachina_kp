# Compute a Gram matrix, all kernel values between sets of datapoints
# U is N-by-R
# V is M-by-R
# K is N-by-M, K(u,v) is kernel(U(u,:), V(v,:))

import numpy as np
from dist2 import dist2

def gramMatrix(U, V, kernel, kernel_params):

  if   kernel == 'gaussian':
    s                                            = kernel_params
    K                                            = np.exp( -         dist2(U, V)  / (2 * s**2) )
  elif kernel == 'exponential':
    s                                            = kernel_params
    K                                            = np.exp( - np.sqrt(dist2(U, V)) / (2 * s**2) )
  elif kernel == 'cauchy':
    s                                            = kernel_params
    K                                            = 1 / (1  +         dist2(U, V)  / (    s**2) )
  elif kernel == 'student':
    d                                            = kernel_params
    K                                            = 1 / (1  + np.sqrt(dist2(U, V)) ** d)
  elif kernel == 'power':
    d                                            = kernel_params
    K                                            =         - np.sqrt(dist2(U, V)) ** d
  elif kernel == 'log':
    d                                            = kernel_params
    K                                            =  - np.log(np.sqrt(dist2(U, V)) ** d + 1)
  elif kernel == 'sigmoid':
    mu                                           = kernel_params[0]
    s                                            = kernel_params[1]
    K                                            = 1 / (1 + np.exp( (mu - np.sqrt(dist2(U, V))) / s ))
  else:
    print               'Unknown kernel type'

  return K
