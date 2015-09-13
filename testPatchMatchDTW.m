N = 1000;
M = 1200;
t1 = linspace(0, 1, N);
t2 = linspace(0, 1, M).^3;
t1 = 2*pi*t1;
t2 = 2*pi*t2;

X1 = [cos(t1(:)) sin(t1(:))];
X2 = [cos(t2(:)) sin(t2(:))];
NNFunction = @(x, y) pdist2(x(:)', y(:)'); %Test with L2 (dim is dummy variable in this case)
NIters = 5;
K = 2;
DOPLOT = 0;

DGT = pdist2(X1, X2);

[NNF, Queries] = patchMatch1DMatlab( X1, X2, NNFunction, NIters, K, DOPLOT );
S1 = 1:size(NNF, 1);

subplot(1, 2, 1);
imagesc(DGT);
title('Ground Truth Pairwise Distance Function');
colormap('jet');
subplot(1, 2, 2);
DOut = sparse(repmat(1:N, [1, K])', NNF(:), ones(N*K, 1), N, M);
imagesc(DOut);
title('Patch Match Answer');
colormap('default');