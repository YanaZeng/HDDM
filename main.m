function [results] = main(data,skeleton,nsamp) %nsamp skeletonNames
% Multivariate nonlinear deterministic cases

%INPUT
%  X(alternatively): cell. Each element of X is a matrix with dims*samples.

%OUTPUT
%  results:
%    .order: causal order

% skeletonNames = 'asia'; % real dataset-stock market data stock
% %generate data
% if strcmp(skeletonNames, 'stock')
%     data_raw = csvread('F:\Dropbox\Dropbox\Yana_enlight\1-deterministic\Codes-deterministic\stock-dataset\stock_012_final-China.csv'); 
%     data_raw =  transpose(data_raw);
%     data = cell(1,3); %America1;Europe2;Asia3
%     data{1} = data_raw(1:4,:);
%     data{2} = data_raw([6,7],:);
%     data{3} = data_raw(11:12,:);  
% else
%     [data,skeleton] = generate_data_real(skeletonNames,nsamp); %real network
% end


 

nsamp = size(data{1},2);
X = data;

n = length(X); % Number of variables
T = size(X{1},2); % Sample size
lam = 1e-5; %lambda;

%normalize the data X
for i = 1:n
    tmp = X{i}';%note tmp is samples*dims
    tmp=tmp-repmat(mean(tmp),size(tmp,1),1);%mean(tmp,2):每一行的均值;mean(tmp):每一列
    tmp=tmp*diag(1./std(tmp));
    X{i} = tmp';
end

%compute all the Gram matrix
Gram = cell(1,n);
for i = 1:n
    xx = X{i}';
    % set the kernel for X
    GX = sum((xx.*xx),2); %xx is samples*dims
    Q = repmat(GX,1,T);
    R = repmat(GX',T,1);
    dists = Q + R - 2*xx*xx';
    dists = dists-tril(dists); %tril(X): X's lower triangular matrix
    dists = reshape(dists,T^2,1); %returns the M-by-N matrix whose elements are taken columnwise from X
    width = median(dists(dists>0))/nsamp; %5.4kernel
%     width = median(dists(dists>0))/9;%9    sqrt(0.2*median(dists(dists>0))); % median value
%     width = width*2; %%% width = 2*median。
    theta = 1/(width^2);
    
    Kx = kernel([xx], [xx], [theta,1]); % Gaussian kernel   theta/2 等价于 1/(2*width^2)    ;
    
    H0 =  eye(T) - ones(T,T)/(T); % for centering of the data in feature space
    Kx = H0 * Kx * H0; % kernel matrix for X
    
    Gram{i} = Kx;
end


%iteratively select the order
Matrix_D = NaN(n,n);
K = zeros(1,n);

for k = 1:n-1
    if k ==1 %select the first root
        for i = 1:n
            for j = 1:n
                if i ~= j
                    %calculate the estimate of R, AA, S and ASA.
                    Ki = Gram{i};
                    R = X{j}*pdinv(Ki+lam*eye(T));%m*T
                    AA = Ki*R'*R;
                    S = Ki;
                    ASA = Ki^2*R'*R;
                    
                    %calculate difference, i.e log(tao(ASA))-log(tao(S))-log(tao(AA))
                    D = lnt(ASA)-lnt(S)-lnt(AA);
                    Matrix_D(i,j) = D;
                end
            end
        end
        
        %method to select the root
        select = 2;
        %0:choose the min(abs+sum);
        %1:choose the min(max(abs));
        %2:compare ij and ji and set the smaller be 1, choose max()
        Matrix_D_abs = abs(Matrix_D);%Matrix_D
        
        %delete the non-deterministic cases, i.e. (ij)(ji)are too close. nd
        nd_ind = [];
        for i = 1:n-1
            for j = i+1:n
                if abs(Matrix_D_abs(i,j)-Matrix_D_abs(j,i))<=0.25
                    nd_ind = [nd_ind;i,j];
                    Matrix_D_abs(i,j) = 0;
                    Matrix_D_abs(j,i) = 0;
                end
            end
        end
        
        %store some absolute deterministic cases, ad
        ad_matrix = zeros(n,n);
        [ad_row,ad_col] = find(Matrix_D_abs<1);%0.2
        for i = 1:length(ad_row)
            if Matrix_D_abs(ad_col(i),ad_row(i))>2
                ad_matrix(ad_col(i),ad_row(i))=1;
                ad_matrix(ad_row(i),ad_col(i))=1;
            end
        end
        [ad_ind1,ad_ind2] = find(ad_matrix~=0);
        
        
        %if exist 0 in Matrix_D_abs, update it.
        if ~isempty(find(Matrix_D_abs(:)==0, 1))
            [update_ind1,update_ind2] = find(Matrix_D_abs==0);
            for i = 1:length(update_ind1)
                X_Pa = X{update_ind1(i)};
                Pa_candi = setdiff([1:n],[update_ind1(i),update_ind2(i)]);
                for l = 1:n-2 %add the data of the selected roots to X_Pa
                    X_Pa_ind = Pa_candi(l);
                    X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                end
                
                %calculate Gram of X_Pa
                X_Pa = X_Pa';
                Gram_X_Pa = cal_Gram(X_Pa,nsamp);
                
                %calculate the estimate of R, AA, S and ASA.
                Ki = Gram_X_Pa;
                R = X{update_ind2(i)}*pdinv(Ki+lam*eye(T));%m*T
                AA = Ki*R'*R;
                S = Ki;
                ASA = Ki^2*R'*R;
                
                %calculate difference, i.e log(tao(ASA))-log(tao(S))-log(tao(AA))
                D = lnt(ASA)-lnt(S)-lnt(AA);
                Matrix_D_abs(update_ind1(i),update_ind2(i)) = abs(D);
            end
        end
        
        if select == 0
            root_val_cand = sum(Matrix_D_abs,2);
            [root_val,root_ind] = min(root_val_cand);
        elseif select == 1
            root_val_cand = max(Matrix_D_abs,[],2);
            [root_va,root_ind] = min(root_val_cand);
        elseif select ==2
            Matrix_D_01 = zeros(n,n);
            for i = 1:n
                for j = i+1:n
                    if Matrix_D_abs(i,j) < Matrix_D_abs(j,i)
                        Matrix_D_01(i,j) = 1;
                    elseif Matrix_D_abs(i,j) > Matrix_D_abs(j,i)
                        Matrix_D_01(j,i) = 1;
                    end
                end
            end
            %Matrix_D_01
            root_val_cand = sum(Matrix_D_01,2);
            if length(find(root_val_cand==max(root_val_cand)))>1
                many_roots = find(root_val_cand==max(root_val_cand));
                for i = 1:length(find(root_val_cand==max(root_val_cand)))
                    for j = i+1:length(find(root_val_cand==max(root_val_cand)))
                        if ad_matrix(many_roots(i),many_roots(j))==1
                            root_ind = many_roots(i);
                            break;
                        elseif ad_matrix(many_roots(j),many_roots(i))==1
                            root_ind = many_roots(j);
                            break;
                        end
                    end
                end
            end
            if ~exist('root_ind','var')
                [root_va,root_ind] = max(root_val_cand);
            end
        end
        K(k) = root_ind;
        
    else %select the second or more root(s)
        %let values of the selected root in Matrix_D be zeros
        Matrix_D_abs(K(k-1),:) = NaN;
        Matrix_D_abs(:,K(k-1)) = NaN;
        
        for i = 1:n-1
            for j = i+1:n
                if i ~= j && ~ismember(i,K) && ~ismember(j,K) && ad_matrix(i,j)==0
                    %-> direction:
                    X_Pa = X{i};
                    for l = 1:sum(K~=0) %add the data of the selected roots to X_Pa
                        if ad_matrix(K(l),i)==0 && ad_matrix(K(l),j)==0 %ignore the other parent if existed
                            if l==1
                                X_Pa_ind = K(l);
                                X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                            else
                                kk = 1:l-1;
                                if all(ad_matrix(K(kk),K(l))==0)
                                    X_Pa_ind = K(l);
                                    X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                                else
                                    [ind11,ind22] = find(ad_matrix(K(kk),K(l))~=0);
                                    if all(ad_matrix(K(ind11),i)==0) && all(ad_matrix(K(ind11),j)==0)
                                        X_Pa_ind = K(l);
                                        X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                                    end
                                end
                            end
                        end
                    end
                    
                    %calculate Gram of X_Pa
                    X_Pa = X_Pa';
                    Gram_X_Pa = cal_Gram(X_Pa,nsamp);
                    
                    %calculate the estimate of R, AA, S and ASA.
                    Ki = Gram_X_Pa;
                    R = X{j}*pdinv(Ki+lam*eye(T));%m*T
                    AA = Ki*R'*R;
                    S = Ki;
                    ASA = Ki^2*R'*R;
                    
                    %calculate difference, i.e log(tao(ASA))-log(tao(S))-log(tao(AA))
                    D = lnt(ASA)-lnt(S)-lnt(AA);
                    D1 = D;
                    
                    %<- direction:
                    X_Pa = X{j};
                    for l = 1:sum(K~=0) %add the data of the selected roots to X_Pa
                        if ad_matrix(K(l),i)==0 && ad_matrix(K(l),j)==0 %ignore the other parent if existed
                            if l==1
                                X_Pa_ind = K(l);
                                X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                            else
                                kk = 1:l-1;
                                if all(ad_matrix(K(kk),K(l))==0)
                                    X_Pa_ind = K(l);
                                    X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                                else
                                    [ind11,ind22] = find(ad_matrix(K(kk),K(l))~=0);
                                    if all(ad_matrix(K(ind11),i)==0) && all(ad_matrix(K(ind11),j)==0)
                                        X_Pa_ind = K(l);
                                        X_Pa = cat(1,X_Pa,X{X_Pa_ind});
                                    end
                                end
                            end
                        end
                    end
                    
                    %calculate Gram of X_Pa
                    X_Pa = X_Pa';
                    Gram_X_Pa = cal_Gram(X_Pa,nsamp);
                    
                    %calculate the estimate of R, AA, S and ASA.
                    Kj = Gram_X_Pa;
                    R = X{i}*pdinv(Kj+lam*eye(T));%m*T
                    AA = Kj*R'*R;
                    S = Kj;
                    ASA = Kj^2*R'*R;
                    
                    %calculate difference, i.e log(tao(ASA))-log(tao(S))-log(tao(AA))
                    D = lnt(ASA)-lnt(S)-lnt(AA);
                    D2 = D;
                    
                    if abs(D2-D1)>abs(Matrix_D_abs(i,j)-Matrix_D_abs(j,i))
                        Matrix_D_abs(i,j) = abs(D1);
                        Matrix_D_abs(j,i) = abs(D2);
                    end
                end
            end
        end
        
        %method to select the root
        select = 2;
        %0:choose the min(abs+sum);
        %1:choose the min(max(abs));
        %2:compare ij and ji and set the smaller be 1, choose max()
        %         Matrix_D_abs = abs(Matrix_D);
        
        %delete the non-deterministic cases, i.e. (ij)(ji)are too close. nd
        nd_ind = [];
        for i = 1:n-1
            for j = i+1:n
                if abs(Matrix_D_abs(i,j)-Matrix_D_abs(j,i))<=0.15
                    nd_ind = [nd_ind;i,j];
                    Matrix_D_abs(i,j) = 0;
                    Matrix_D_abs(j,i) = 0;
                end
            end
        end
        
        if select == 0
            root_val_cand = sum(Matrix_D_abs,2);
            [root_val,root_ind] = min(root_val_cand);
        elseif select == 1
            root_val_cand = max(Matrix_D_abs,[],2);
            [root_va,root_ind] = min(root_val_cand);
        elseif select ==2
            Matrix_D_01 = zeros(n,n);
            for i = 1:n
                for j = i+1:n
                    if Matrix_D_abs(i,j) < Matrix_D_abs(j,i)
                        Matrix_D_01(i,j) = 1;
                    elseif Matrix_D_abs(i,j) > Matrix_D_abs(j,i)
                        Matrix_D_01(j,i) = 1;
                    else Matrix_D_abs(i,j) = Matrix_D_abs(j,i); 
                        if ~isnan(Matrix_D_abs(j,i))
                        Matrix_D_01(j,i) = 0.1;
                        Matrix_D_01(i,j) = 0.1;
                        end
                    end
                end
            end
            root_val_cand = sum(Matrix_D_01,2);
            root_ind = [];
            if length(find(root_val_cand==max(root_val_cand)))>1
                many_roots = find(root_val_cand==max(root_val_cand));
                for i = 1:length(find(root_val_cand==max(root_val_cand)))
                    for j = i+1:length(find(root_val_cand==max(root_val_cand)))
                        if ad_matrix(many_roots(i),many_roots(j))==1
                            root_ind = many_roots(i);
                            break;
                        elseif ad_matrix(many_roots(j),many_roots(i))==1
                            root_ind = many_roots(j);
                            break;
                        end
                    end
                end
            end
            if isempty(root_ind)
                [root_va,root_ind] = max(root_val_cand);
            end
        end
        K(k) = root_ind;
    end
end
last_ind = setdiff([1:n],K);
K(n) = last_ind;
results.order = K;
K

%evaluate the order
[Recall,Precision,F1,First] = Analysis_kest(skeleton,K)
results.order_Recall = Recall;
results.order_Precision = Precision;
results.order_F1 = F1;
results.order_First = First;

% %prune acc. to order
% g = pc(data,K,'indtest_KCI',[],2,0.05);



end



%lnt function which helps to calculate log(tao(A)), A is a matrix.
function [value] = lnt(A)

tao = sum(diag(A))/size(A,1);
value = log(tao); %log=ln

end

%calculate Gram matrix using data(samples*dims)
function [Kx] = cal_Gram(xx,nsamp)
% function [Kx] = cal_Gram(xx)
T = size(xx,1);
% set the kernel for X
GX = sum((xx.*xx),2); %xx is samples*dims
Q = repmat(GX,1,T);
R = repmat(GX',T,1);
dists = Q + R - 2*xx*xx';
dists = dists-tril(dists); %tril(X): X's lower triangular matrix
dists=reshape(dists,T^2,1); %returns the M-by-N matrix whose elements are taken columnwise from X
width = median(dists(dists>0))/nsamp;%5.4kernel width
% width = median(dists(dists>0))/9;%sqrt(0.5*median(dists(dists>0))); %median value
% width = width*2; %%%
theta = 1/(width^2);

Kx = kernel([xx], [xx], [theta,1]); % Gaussian kernel

H0 =  eye(T) - ones(T,T)/(T); % for centering of the data in feature space
Kx = H0 * Kx * H0; % kernel matrix for X
end


