% %A slow implementation of generalized patch match in Matlab for testing purposes
% %NNFunction is a function which takes two patches and a patch dimension
% %NIters: Number of iterations (not many are needed before convergence)
% %K: Number of nearest neighbors to consider in the nearest neighbor field
% %(for reshaping purposes) and returns a distance
function [ NNF, Queries, DOut ] = patchMatch1DMatlab( X1, X2, NNFunction, NIters, K, DOPLOT )
    SwitchOddEven = 0;
    Queries = 0;
        
    N = size(X1, 1);
    M = size(X2, 1);
    
    Queried = sparse(N, M);%Keep track of distances that are already queried
    %so that no work is redone (TODO: Make this sparse?)
    
    %Randomly initialize nearest neighbor field
    NNF = zeros(N, K);
    for ii = 1:N
        NNF(ii, :) = randperm(M, K);
    end
    %Bias towards the diagonal
%     NNF = repmat((1:M)', [1 K]);
    DNNF = zeros(N, K);
    alpha = 0.5;
    for ii = 1:N
        if N > 1000 && mod(ii, 1000) == 0
            fprintf(1, '.');
        end
        for kk = 1:K
            DNNF(ii, kk) = NNFunction(X1(ii, :), X2(NNF(ii, kk), :));
            %Queried(ii, NNF(ii, kk)) = 1;
        end
    end
    num = 0;
    for iter = 1:NIters
        fprintf(1, 'iter = %i\n', iter);
        for ii = 1:N
            if DOPLOT
                DOut = sparse(repmat(1:N, [1, K])', NNF(:), ones(N*K, 1), N, M);
                clf;
                imagesc(DOut);
                hold on;
                plot([0 size(DOut, 2)], [ii ii], 'r');
                title(sprintf('Iteration %i %.3g%s Queries', iter, Queries/(N*M)*100, '%'));
                print('-dpng', '-r100', sprintf('%i.png', num));
                num = num + 1;
            end
            if N > 1000 && mod(ii, 1000) == 0
                fprintf(1, '.');
            end
            %STEP 1: Propagate
            idx = ii;%Index of current pixel
            di = -1;
            if mod(iter, 2) == 0 && SwitchOddEven %On even iterations propagate the other way
                idx = N - ii + 1;
                di = 1;
            end
            if ii > 1
                indices = [NNF(ii, :) zeros(1, K)];
                dists = [DNNF(ii, :) inf*ones(1, K)];
                for kk = 1:K
                    otherM = NNF(idx + di, kk) - di;
                    if otherM < 1 || otherM > M %Bounds check
                        continue;
                    end
                    if Queried(ii, otherM) %Don't repeat work
                        continue;
                    end
                    Queried(ii, otherM) = 1;
                    Queries = Queries + 1;
                    indices(K+kk) = otherM;
                    dists(K+kk) = NNFunction(X1(idx, :), X2(otherM, :));
                end
                %Pick the top K neighbors out of the K old ones and 
                %the K new ones
                [dists, distsorder] = sort(dists);
                keep = ones(size(dists));
                for kk = 2:length(keep)
                    if dists(kk) == dists(kk-1)
                        keep(kk) = 0;
                    end
                end
                dists = dists(keep == 1);
                indices = indices(distsorder(keep == 1));
                NNF(ii, :) = indices(1:K);
                DNNF(ii, :) = dists(1:K);
            end
            %STEP 2: Random search
            Ri = M*(2*rand(1) - 1);
            radii = [];
            jj = 1;
            while abs(round(Ri*alpha^jj)) > 1
                radii = [radii int32(round(Ri*alpha^jj))];
                jj = jj + 1;
            end
            NR = length(radii);
            if NR == 0
                continue;
            end
            indices = [NNF(ii, :) zeros(1, K*NR)];
            dists = [DNNF(ii, :) inf*ones(1, K*NR)];
            for rr = 1:NR
                for kk = 1:K
                    otherM = radii(rr) + NNF(idx, kk);
                    if otherM < 1 || otherM > M %Bounds check
                        continue;
                    end
                    if Queried(ii, otherM) %Don't repeat work
                        continue;
                    end
                    Queried(ii, otherM) = 1;
                    Queries = Queries + 1;
                    indices(K+(rr-1)*K+kk) = otherM;
                    dists(K+(rr-1)*K+kk) = NNFunction(X1(idx, :), X2(otherM, :));
                end
            end
            %Pick the top K neighbors out of the K old ones and 
            %the K new ones
            [dists, distsorder] = sort(dists);
            keep = ones(size(dists));
            for kk = 2:length(keep)
                if dists(kk) == dists(kk-1)
                    keep(kk) = 0;
                end
            end
            dists = dists(keep == 1);
            indices = indices(distsorder(keep == 1));
            NNF(ii, :) = indices(1:K);
            DNNF(ii, :) = dists(1:K);
        end
    end
end
