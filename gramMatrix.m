function K                                       = gramMatrix(U, V, kernel, kernel_params)
% K = gramMatrix(U, V, 'gaussian', {sigma})
%
% Compute a Gram matrix, all kernel values between sets of datapoints.
% U is N-by-R
% V is M-by-R
% K is N-by-M, K(u,v) is kernel(U(u,:), V(v,:))

if     strcmp(kernel, 'gaussian')
  s                                              = kernel_params{1};
  K                                              = exp( -      dist2(U, V)  / (2 * s^2) );
elseif strcmp(kernel, 'exponential')
  s                                              = kernel_params{1};
  K                                              = exp( - sqrt(dist2(U, V)) / (2 * s^2) );
elseif strcmp(kernel, 'cauchy')
  s                                              = kernel_params{1};
  K                                              = 1 ./ (1 +      dist2(U, V)   / (s^2));
elseif strcmp(kernel, 'student')
  d                                              = kernel_params{1};
  K                                              = 1 ./ (1 + sqrt(dist2(U, V)) .^ d);
elseif strcmp(kernel, 'power')
  d                                              = kernel_params{1};
  K                                              = -     sqrt(dist2(U, V)) .^ d;
elseif strcmp(kernel, 'log')
  d                                              = kernel_params{1};
  K                                              = - log(sqrt(dist2(U, V)) .^ d + 1);
elseif strcmp(kernel, 'sigmoid')
  mu                                             = kernel_params{1};
  s                                              = kernel_params{2};
  K                                              = 1 ./ (1 + exp(   (   repmat(mu, [size(U, 1), size(V, 1)]) ...
                                                                      - sqrt(dist2(U, V))   ) ...
                                                                  / s   ));
elseif strcmp(kernel, 'histinter')
  size_U                                         = size(U, 1);
  size_V                                         = size(V, 1);
  K                                              = zeros(size_U, size_V);
  for i = 1:size_U
    if mod(i, 200) == 0
      fprintf('i: %d\n', i);
    end
    for j = 1:size_V
      K(i, j)                                    = sum(min(U(i, :), V(j, :)));
    end
  end
else
  error('Unknown kernel type');
end

% Gaussian
% Training misclassification typically 1 (but up to ~10-20 range), testing Ham/Spam ~70/30.
% 2  - tr. miscl.  ~1,    test Ham/Spam ~70/30
% 5  - tr. miscl.  ~1,    test Ham/Spam ~70/30
% 10 - tr. miscl.  ~5,    test Ham/Spam ~70/30
% 20 - tr. miscl. ~10-30, test Ham/Spam ~65/35
% 50 - tr. miscl. ~10,    test Ham/Spam ~65/35

% Exponential
% 2  - tr. miscl.  ~2,    test Ham/Spam ~65/35
% 5  - tr. miscl.  ~5,    test Ham/Spam ~60/40
% 10 - tr. miscl.  ~1,    test Ham/Spam ~60/40
% 20 - tr. miscl. ~10-60, test Ham/Spam ~75/25
% 50 - tr. miscl.  ~5-80, test Ham/Spam ~70/30

% Cauchy
% 2,5,10 - Training miscl. ~1,    testing Ham/Spam ~70/30 (i.e. very similar to Gaussian)
% 20     - Training miscl. ~3,    testing Ham/Spam ~60/40
% 50     - Training miscl. ~8-30, testing Ham/Spam ~60/40
% Also tried 0.1,0.5,1, similar to 2,5,10, certainly no better

% T-Student
%  2 - Training miscl. ~1,    testing Ham/Spam ~70/30
%  5 - Training miscl. ~1-10, testing Ham/Spam ~95/5
% 10 - Training miscl. ~1-3,  testing Ham/Spam ~95/5 ~80/20
% 20 - Training miscl. ~5,    testing Ham/Spam ~90/10
% 50 - Training miscl. ~5,    testing Ham/Spam ~80/20

% Power
% 2,5,10 train Ham/Spam = 0/100, test Ham/Spam = 0/100
% 20,50  train Ham/Spam = 0/100, test Ham/Spam = 0/95 (i.e. NaNs)

% Log
% 2,5,10 train Ham/Spam = 0/100, test Ham/Spam = 0/100
% 20,50  train Ham/Spam = 0/100, test Ham/Spam = 0/95 (i.e. NaNs)

% Sigmoid

% Histogram intersection
% Train Ham/Spam = 0/100, testing Ham/spam = 0/100
