#  DIST2 Calculates squared distance between two sets of points.
#
#  Description
#  D = DIST2(X, C) takes two matrices of vectors and calculates the
#  squared Euclidean distance between them.  Both matrices must be of
#  the same column dimension.  If X has M rows and N columns, and C has
#  L rows and N columns, then the result has M rows and L columns.  The
#  I, Jth entry is the  squared distance from the Ith row of X to the
#  Jth row of C.
#
#  See also
#  GMMACTIV, KMEANS, RBFFWD
#

#  Copyright (c) Ian T Nabney (1996-9)

import numpy as np

def dist2(x, c):

  ndata,    dimx                                 = np.shape(x)
  ncentres, dimc                                 = np.shape(c)
  if dimx != dimc:
    print                 'Data dimension does not match dimension of centres'

  n2                                             =   np.transpose(np.repeat(
                                                                    np.sum(
                                                                      np.transpose(x ** 2),
                                                                      axis = 0)[np.newaxis, :],
                                                                    ncentres, 0))                                           \
                                                   +              np.repeat(
                                                                    np.sum(
                                                                      np.transpose(c ** 2),
                                                                      axis = 0)[np.newaxis, :],
                                                                    ndata   , 0)                                            \
                                                   - 2 * np.dot(x, np.transpose(c))

# Rounding errors occasionally cause negative entries in n2:
  if True in (n2 < 0):
    n2[np.where(n2 < 0)]                         = 0

  return n2
