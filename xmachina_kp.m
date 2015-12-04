% xmachina_kp.m
% Darren Temple

% --------------------------------------------------------------------------------------------------------------------------
% Variables
% --------------------------------------------------------------------------------------------------------------------------

normalise                                        = false; % Works better with false
verbose                                          = false;

N_iter                                           = 100;     % Default 100
eta                                              = 0.00001; % Default 0.00001

find_best                                        = false;
kernel_parameters_best                           = 4.5; % Cauchy
%kernel_parameters_best                           = 5; % Gaussian

% Kernel parameters:
%kernel_type                                      = 'gaussian';
%kernel_parameters                                = 5;
%kernel_type                                      = 'exponential';
%kernel_parameters                                = 2;
kernel_type                                      = 'cauchy';
kernel_parameters                                = 0.5:0.5:5;
%kernel_type                                      = 'student';
%kernel_parameters                                = 0.5:0.5:5;
%kernel_type                                      = 'power';
%kernel_parameters                                = 2;
%kernel_type                                      = 'log';
%kernel_parameters                                = 2;
%kernel_type                                      = 'sigmoid';
%kernel_parameters                                = {5, 5};
%kernel_type                                      = 'histinter';
%kernel_parameters                                = 1; % Dummy, as no parameters for this kernel type
size_kernel_parameters                           = size(kernel_parameters, 2);

% --------------------------------------------------------------------------------------------------------------------------

% Load the training data:
filename                                         = '../training_10000events.csv';
file                                             = fopen(filename);
file_full_linear                                 = textscan(file, '%s', 'Delimiter', ',');
fclose(file);

% Extract feature names and rest of data:
file_ncol                                        = 33;
file_full                                        = reshape(file_full_linear{1}, file_ncol, [])';
features                                         = file_full(1, 2:file_ncol-2)';
data                                             = str2double(file_full(2:end, 2:file_ncol-2));
t                                                = cell2mat(  file_full(2:end, end));
% Map from the given t = {'s', 'b'} to t = {+1, -1}:
t                                                =   (t == 's') ...
                                                   - (t ~= 's');
[N, D]                                           = size(data);
clear file_full;

N_train                                          = 5000;
N_test                                           = N - N_train;

data_train                                       = data(        1:N_train, :);
data_test                                        = data(N_train+1:end    , :);
clear data;

t_train                                          =    t(        1:N_train, :);
t_test                                           =    t(N_train+1:end    , :);
clear t;

%for shuffle = 1:5
%  randperm_N_train                               = randperm(N_train);
%  data_train                                     = data_train(randperm_N_train, :);
%  t_train                                        = t_train(   randperm_N_train);
%end

if normalise
  mean_data_train                                = mean(data_train, 1);
  std_data_train                                 = std( data_train, 1, 1) + eps;
  data_train                                     = data_train  - repmat(mean_data_train, [N_train, 1]);
  data_train                                     = data_train ./ repmat( std_data_train, [N_train, 1]);
  data_test                                      = data_test   - repmat(mean_data_train, [N_test , 1]);
  data_test                                      = data_test  ./ repmat( std_data_train, [N_test , 1]);
end

% --------------------------------------------------------------------------------------------------------------------------

data_train_rowindex_all                          = 1:N_train;

N_per_valset                                     = 200;
N_valset                                         = N_train / N_per_valset;

alpha                                            = zeros(N_train, 1);
N_incorrect_train_val                            = zeros(size_kernel_parameters, N_valset);

% --------------------------------------------------------------------------------------------------------------------------
% Calculation
% --------------------------------------------------------------------------------------------------------------------------

% Train kernel perceptron:

if find_best

% Find the best kernel parameter(s):

  fprintf               ('\nFinding the best kernel parameter(s) ...\n\n');

  for kernel_parameters_index = 1:size_kernel_parameters

    fprintf             ('Parameter: %d/%d\n', kernel_parameters_index, size_kernel_parameters);

    K_train                                      = gramMatrix(data_train, data_train, kernel_type, ...
                                                             {kernel_parameters(kernel_parameters_index)});

    for valset = 1:N_valset

      fprintf           ('Valset: %d/%d\n', valset, N_valset);

      data_train_rowindex_val                    = [1:N_per_valset] + N_per_valset * (valset - 1);
      data_train_rowindex_use                    = setdiff(data_train_rowindex_all, data_train_rowindex_val);

% Determine alpha for the current use set:

      for iter = 1:N_iter

        fprintf         ('iter: %d/%d\n', iter, N_iter);

        for data_train_use_rowindex = randperm(N_train - N_per_valset)

          y_train_use_current                    = sign(   alpha(  data_train_rowindex_use)' ...
                                                         * K_train(data_train_rowindex_use, ...
                                                                   data_train_rowindex_use(data_train_use_rowindex))   );

          if y_train_use_current ~= t_train(data_train_rowindex_use(data_train_use_rowindex))
            alpha(data_train_rowindex_use(data_train_use_rowindex)) ...
                                                 =   alpha(        data_train_rowindex_use(data_train_use_rowindex)) ...
                                                   + eta * t_train(data_train_rowindex_use(data_train_use_rowindex));
          end

        end

        y_train_use                              = sign(   alpha(  data_train_rowindex_use)' ...
                                                         * K_train(data_train_rowindex_use, ...
                                                                   data_train_rowindex_use))';
        y_train_use_index_zero                   = (y_train_use == 0);

        if sum(y_train_use_index_zero) ~= 0
          y_train_use(y_train_use_index_zero)    = 1;
        end

        N_incorrect_train_use                    = sum(y_train_use ~= t_train(data_train_rowindex_use));

        if verbose
          fprintf       ('N_incorrect_train_use: %d\n', N_incorrect_train_use);
        end

        if N_incorrect_train_use == 0
          break
        end

      end % for iter

% Try alpha with the current validation set:

      y_train_val                                = sign(   alpha(  data_train_rowindex_use)' ...
                                                         * K_train(data_train_rowindex_use, ...
                                                                   data_train_rowindex_val))';
      y_train_val_index_zero                     = (y_train_val == 0);

      if sum(y_train_val_index_zero) ~= 0
        y_train_val(y_train_val_index_zero)      = 1;
      end

      N_incorrect_train_val(kernel_parameters_index, valset) ...
                                                 = sum(y_train_val ~= t_train(data_train_rowindex_val));

      alpha                                      = zeros(N_train, 1);

    end % for valset

  end % for kernel_parameters_index

  sum_N_incorrect_train_val                      = sum(     N_incorrect_train_val, 2);
  min_sum_N_incorrect_train_val                  = min( sum_N_incorrect_train_val);
  kernel_parameters_index_best                   = find(sum_N_incorrect_train_val == min_sum_N_incorrect_train_val);
  kernel_parameters_best                         = kernel_parameters(kernel_parameters_index_best(1));

end

% --------------------------------------------------------------------------------------------------------------------------

% Retrain on the full dataset:

fprintf                 ('\nTraining ...\n\n');

if size_kernel_parameters > 1
  K_train                                        = gramMatrix(data_train, data_train, kernel_type, ...
                                                             {kernel_parameters_best});
end

for iter = 1:N_iter

  fprintf               ('iter: %d/%d\n', iter, N_iter);

  for data_train_rowindex = randperm(N_train)

    y_train_use_current                          = sign(alpha' * K_train(:, data_train_rowindex));

    if y_train_use_current ~= t_train(data_train_rowindex)
      alpha(data_train_rowindex)                 =   alpha(        data_train_rowindex) ...
                                                   + eta * t_train(data_train_rowindex);
    end

  end

  y_train                                        = sign(alpha' * K_train)';
  y_train_index_zero                             = (y_train == 0);

  if sum(y_train_index_zero) ~= 0
    y_train(y_train_index_zero)                  = 1;
  end

  N_incorrect_train                              = sum(y_train ~= t_train);

  if verbose
    fprintf             ('N_incorrect_train: %d\n', N_incorrect_train);
  end

  if N_incorrect_train == 0
    break
  end

end

% --------------------------------------------------------------------------------------------------------------------------

% Test kernel perceptron:

K_test                                           = gramMatrix(data_train, data_test, kernel_type, ...
                                                             {kernel_parameters_best});

% Try alpha with the test set:

y_test                                           = sign(alpha' * K_test)';
y_test_index_zero                                = (y_test == 0);

if sum(y_test_index_zero) ~= 0
  y_test(y_test_index_zero)                      = 1;
end

N_incorrect_test                                 = sum(y_test ~= t_test);

% Determine signal and background counts:

N_sig_t_train                                    = sum(t_train ==  1);
N_bkg_t_train                                    = sum(t_train == -1);
N_sig_t_test                                     = sum(t_test  ==  1);
N_bkg_t_test                                     = sum(t_test  == -1);

N_sig_y_train                                    = sum(y_train ==  1);
N_bkg_y_train                                    = sum(y_train == -1);
N_sig_y_test                                     = sum(y_test  ==  1);
N_bkg_y_test                                     = sum(y_test  == -1);

if (N_sig_y_train + N_bkg_y_train) ~= N_train
  fprintf               ('ERROR: (N_sig_y_train + N_bkg_y_train) ~= N_train\n');
end

if (N_sig_y_test + N_bkg_y_test) ~= N_test
  fprintf               ('ERROR: (N_sig_y_test + N_bkg_y_test) ~= N_test\n');
end

% --------------------------------------------------------------------------------------------------------------------------
% Output
% --------------------------------------------------------------------------------------------------------------------------

fprintf                 ('\n')
fprintf                 ('     Kernel type      : %s\n', kernel_type)
fprintf                 ('     kernel parameters: %d\n', kernel_parameters_best)
fprintf                 ('\n')
fprintf                 ('     Training: Total: %4d\n', N_train)
fprintf                 ('               N_sig: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_sig_y_train, (N_sig_y_train / N_train) * 100, N_sig_t_train, (N_sig_t_train  / N_train) * 100);
fprintf                 ('               N_bkg: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_bkg_y_train, (N_bkg_y_train / N_train) * 100, N_bkg_t_train, (N_bkg_t_train  / N_train) * 100);
fprintf                 ('       Misclassified: %4d (%6.2f%%)\n', N_incorrect_train, (N_incorrect_train / N_train) * 100);
fprintf                 ('\n')
fprintf                 ('     Testing : Total: %4d\n', N_test)
fprintf                 ('               N_sig: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_sig_y_test , (N_sig_y_test  / N_test ) * 100, N_sig_t_test , (N_sig_t_test   / N_test ) * 100);
fprintf                 ('               N_bkg: %4d (%6.2f%%)   [Target: %4d (%6.2f%%)]\n', ...
                         N_bkg_y_test , (N_bkg_y_test  / N_test ) * 100, N_bkg_t_test , (N_bkg_t_test   / N_test ) * 100);
fprintf                 ('       Misclassified: %4d (%6.2f%%)\n', N_incorrect_test , (N_incorrect_test  / N_test ) * 100);
fprintf                 ('\n')
