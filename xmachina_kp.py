# xmachina_kp.py
# Darren Temple

import csv
import numpy as np
import random
import sys
from gramMatrix import gramMatrix

# --------------------------------------------------------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------------------------------------------------------

normalise                                        = False # Works better with false
verbose                                          = True
superverbose                                     = False

N_epoch                                          = 100     # Default 100
eta                                              = 0.00001 # Default 0.00001

find_best                                        = True
#kernel_parameters_best                           = 4.5 # Cauchy
#kernel_parameters_best                           = 5.0 # Gaussian

#kernel_type                                      = 'gaussian'
#kernel_parameters                                = [5]
#kernel_type                                      = 'exponential'
#kernel_parameters                                = [2]
kernel_type                                      = 'cauchy'
kernel_parameters                                = [i/10.0 for i in range(5, 51, 5)]
#kernel_type                                      = 'student'
#kernel_parameters                                = [i/10.0 for i in range(5, 51, 5)]
#kernel_type                                      = 'power'
#kernel_parameters                                = [2]
#kernel_type                                      = 'log'
#kernel_parameters                                = [2]
#kernel_type                                      = 'sigmoid'
#kernel_parameters                                = [[5, 5]]
len_kernel_parameters                            = len(kernel_parameters)

# --------------------------------------------------------------------------------------------------------------------------

# Load the training data:

#IFile                                            = open('../training_10000events.csv', 'rU')
#for line in IFile:
#  line_split                                     = line.split(',')
#  data[] = line_split[1:-2]
#csv                                              = np.genfromtxt('../training_100000events.csv', delimiter = ',')

with open('../training_10000events.csv', 'rU') as IFile:
  file_full                                      = [row for row in csv.reader(IFile, delimiter = ',')]

features                                         = file_full[0][1:-2]

D                                                = len(features)
N                                                = len(file_full) - 1
N_train                                          = 5000
N_test                                           = N - N_train

# Map from the given t = {'s', 'b'} to t = {+1, -1}:
data                                             = [map(float, file_full[row][1:-2]) for row in range(1, N+1)]
data                                             = np.array(data)
t                                                = [           file_full[row][  -1]  for row in range(1, N+1)]
t                                                = [1 if i=='s' else -1 for i in t]
t                                                = np.array(t)
del file_full

data_train                                       = data[      0:N_train]
data_test                                        = data[N_train:       ]
del data

t_train                                          =    t[      0:N_train]
t_test                                           =    t[N_train:       ]
del t

#for shuffle = 1:5
#  randperm_N_train                               = randperm(N_train)
#  data_train                                     = data_train(randperm_N_train, :)
#  t_train                                        = t_train(   randperm_N_train)
#end

if normalise:
  mean_data_train                                = np.mean(data_train, axis = 0)
  std_data_train                                 = np.std (data_train, axis = 0)
  data_train                                     = np.subtract(data_train, mean_data_train)
  data_train                                     = np.divide  (data_train,  std_data_train)
  data_test                                      = np.subtract(data_test , mean_data_train)
  data_test                                      = np.divide  (data_test ,  std_data_train)

# --------------------------------------------------------------------------------------------------------------------------

data_train_rowindex_all                          = range(N_train)

N_per_valset                                     = 200 # Default 200
N_valset                                         = N_train / N_per_valset

alpha                                            = np.zeros(N_train)
N_incorrect_train_val                            = np.zeros((len_kernel_parameters, N_valset))

# --------------------------------------------------------------------------------------------------------------------------
# Calculation
# --------------------------------------------------------------------------------------------------------------------------

# Train kernel perceptron:

if find_best:

# Find the best kernel parameter(s):

  if verbose:
    print               '\nFinding the best kernel parameter(s) ...\n'

  for kernel_parameters_index in range(len_kernel_parameters):

    if verbose:
      print             'Parameter: %d/%d' % (kernel_parameters_index + 1, len_kernel_parameters)

    K_train                                      = gramMatrix(data_train, data_train,
                                                              kernel_type, kernel_parameters[kernel_parameters_index])

    for valset in range(N_valset):

      if verbose:
        print           '   Valset: %d/%d' % (valset + 1, N_valset)

      data_train_rowindex_val                    = [i + valset * N_per_valset for i in range(N_per_valset)]
      data_train_rowindex_use                    = range(valset * N_per_valset) + range((valset+1) * N_per_valset, N_train)

# Determine alpha for the current use set:

      for epoch in range(N_epoch):

        if verbose:
          sys.stdout.write('    epoch: %d/%d\r' % (epoch + 1, N_epoch))
          sys.stdout.flush()

        data_train_use_rowindex_shuffle          = range(N_train - N_per_valset)
        random.shuffle(data_train_use_rowindex_shuffle)

        for data_train_use_rowindex in data_train_use_rowindex_shuffle:

          y_train_use_current                    = np.sign(np.dot(  alpha[        data_train_rowindex_use],
                                                                  K_train[np.ix_( data_train_rowindex_use,
                                                                                 [data_train_rowindex_use
                                                                                    [data_train_use_rowindex]] )]  ))[0]

          if y_train_use_current != t_train[data_train_rowindex_use[data_train_use_rowindex]]:
            alpha[data_train_rowindex_use[data_train_use_rowindex]]                                                         \
                                                 = np.add(                   alpha[ data_train_rowindex_use
                                                                                      [data_train_use_rowindex] ],
                                                          np.multiply(eta, t_train[ data_train_rowindex_use
                                                                                      [data_train_use_rowindex] ]))

        # end for data_train_use_rowindex

        y_train_use                              = np.sign(np.dot(  alpha[        data_train_rowindex_use],
                                                                  K_train[np.ix_( data_train_rowindex_use,
                                                                                  data_train_rowindex_use )]  ))

        if 0 in y_train_use:
          y_train_use[np.where(y_train_use == 0)]= 1

        N_incorrect_train_use                    = sum(y_train_use != t_train[data_train_rowindex_use])

        if   superverbose:
          print         '\n    N_incorrect_train_use: %d' % N_incorrect_train_use
        elif verbose & (epoch == (N_epoch - 1)):
          print         ''

        if verbose & (epoch < N_epoch-1) & (N_incorrect_train_use == 0):
          print         '\n    N_incorrect_train_use = 0   =>break'
          break

      # end for epoch

# Try alpha with the current validation set:

      y_train_val                                = np.sign(np.dot(  alpha[        data_train_rowindex_use],
                                                                  K_train[np.ix_( data_train_rowindex_use,
                                                                                  data_train_rowindex_val )]  ))

      if 0 in y_train_val:
        y_train_val[np.where(y_train_val == 0)]  = 1

      N_incorrect_train_val[kernel_parameters_index, valset]                                                                \
                                                 = sum(y_train_val != t_train[data_train_rowindex_val])

      alpha                                      = np.zeros(N_train)

    # end for valset

  # end for kernel_parameters_index

  sum_N_incorrect_train_val                      = np.sum(N_incorrect_train_val, axis = 1)
  kernel_parameters_index_best                   = np.argmin(sum_N_incorrect_train_val)
  kernel_parameters_best                         = kernel_parameters[kernel_parameters_index_best]

# --------------------------------------------------------------------------------------------------------------------------

# Retrain on the full dataset:

if verbose:
  print                 '\nTraining ...\n'

if len_kernel_parameters > 1:
  K_train                                        = gramMatrix(data_train, data_train,
                                                              kernel_type, kernel_parameters_best)

for epoch in range(N_epoch):

  if verbose:
    sys.stdout.write('    epoch: %d/%d\r' % (epoch + 1, N_epoch))
    sys.stdout.flush()

  data_train_rowindex_shuffle                    = range(N_train)
  random.shuffle(data_train_rowindex_shuffle)

  for data_train_rowindex in data_train_rowindex_shuffle:

    y_train_use_current                          = np.sign(np.dot(alpha, K_train[:,data_train_rowindex]))

    if y_train_use_current != t_train[data_train_rowindex]:
      alpha[data_train_rowindex]                 = np.add(                   alpha[data_train_rowindex],
                                                          np.multiply(eta, t_train[data_train_rowindex]))

  # end for data_train_rowindex

  y_train                                        = np.sign(np.dot(alpha, K_train))

  if 0 in y_train:
    y_train[np.where(y_train == 0)]              = 1

  N_incorrect_train                              = sum(y_train != t_train)

  if   superverbose:
    print               '\n    N_incorrect_train: %d' % N_incorrect_train
  elif verbose & (epoch == (N_epoch - 1)):
    print               ''

  if verbose & (epoch < N_epoch-1) & (N_incorrect_train == 0):
    print               '\n    N_incorrect_train = 0   =>break'
    break

# end for epoch

# --------------------------------------------------------------------------------------------------------------------------

# Test kernel perceptron:

K_test                                           = gramMatrix(data_train, data_test,
                                                              kernel_type, kernel_parameters_best)

# Try alpha with the test set:

y_test                                           = np.sign(np.dot(alpha, K_test))

if 0 in y_test:
  y_test[np.where(y_test == 0)]                  = 1

N_incorrect_test                                 = sum(y_test != t_test)

# Determine signal and background counts:

N_sig_t_train                                    = sum(t_train ==  1)
N_bkg_t_train                                    = sum(t_train == -1)
N_sig_t_test                                     = sum(t_test  ==  1)
N_bkg_t_test                                     = sum(t_test  == -1)

N_sig_y_train                                    = sum(y_train ==  1)
N_bkg_y_train                                    = sum(y_train == -1)
N_sig_y_test                                     = sum(y_test  ==  1)
N_bkg_y_test                                     = sum(y_test  == -1)

if (N_sig_y_train + N_bkg_y_train) != N_train:
  print                 'ERROR: (N_sig_y_train + N_bkg_y_train) ~= N_train'

if (N_sig_y_test  + N_bkg_y_test ) != N_test :
  print                 'ERROR: (N_sig_y_test + N_bkg_y_test) ~= N_test'

# --------------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------------

if verbose:
  print                 '\nOutput ...'
print                   ''
print                   '    Kernel type      : %s' % kernel_type
print                   '    kernel parameters: %s' % str(kernel_parameters_best)
print                   ''
print                   '    Training: Total: %4d' % N_train
print                   '              N_sig: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]' %                                   \
                        (N_sig_y_train, float(N_sig_y_train) / float(N_train) * 100,
                         N_sig_t_train, float(N_sig_t_train) / float(N_train) * 100)
print                   '              N_bkg: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]' %                                   \
                        (N_bkg_y_train, float(N_bkg_y_train) / float(N_train) * 100,
                         N_bkg_t_train, float(N_bkg_t_train) / float(N_train) * 100)
print                   '      Misclassified: %4d (%6.2f%%)' %                                                             \
                        (N_incorrect_train, float(N_incorrect_train) / float(N_train) * 100)
print                   ''
print                   '    Testing : Total: %4d' % N_test
print                   '              N_sig: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]' %                                   \
                        (N_sig_y_test , float(N_sig_y_test ) / float(N_test ) * 100,
                         N_sig_t_test , float(N_sig_t_test ) / float(N_test ) * 100)
print                   '              N_bkg: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]' %                                   \
                        (N_bkg_y_test , float(N_bkg_y_test ) / float(N_test ) * 100,
                         N_bkg_t_test , float(N_bkg_t_test ) / float(N_test ) * 100)
print                   '      Misclassified: %4d (%6.2f%%)' %                                                             \
                        (N_incorrect_test , float(N_incorrect_test ) / float(N_test ) * 100)
print                   ''
