i = 1;
j = 3;
abs(Matrix_D_abs(i,j)-Matrix_D_abs(j,i))

%%
i = 4;
j = 5;
X_Pa = X{i};
Pa_candi = [8];
for l = 1:length(Pa_candi) %add the data of the selected roots to X_Pa
    X_Pa_ind = Pa_candi(l);
    X_Pa = cat(1,X_Pa,X{X_Pa_ind});
end

%calculate Gram of X_Pa
X_Pa = X_Pa';
Gram_X_Pa = cal_Gram(X_Pa);

%calculate the estimate of R, AA, S and ASA.
Ki = Gram_X_Pa;
R = X{j}*pdinv(Ki+lam*eye(T));%m*T
AA = Ki*R'*R;
S = Ki;
ASA = Ki^2*R'*R;

%calculate difference, i.e log(tao(ASA))-log(tao(S))-log(tao(AA))
D = lnt(ASA)-lnt(S)-lnt(AA)


%%
nd_indddd = [];
Matrix_D_abssss = Matrix_D_abs;
for i = 1:n
    for j = i+1:n
        if abs(Matrix_D_abs(i,j)-Matrix_D_abs(j,i))<=0.25
            nd_indddd = [nd_indddd;i,j];
            Matrix_D_abssss(i,j) = 0;
            Matrix_D_abssss(j,i) = 0;
        end
    end
end


%%
i=4;
j=5;

%-> direction:
X_Pa = X{i};
for l = 1:sum(K~=0) %add the data of the selected roots to X_Pa
    if ad_matrix(K(l),i)==0 && ad_matrix(K(l),j)==0
        X_Pa_ind = K(l);
        X_Pa = cat(1,X_Pa,X{X_Pa_ind});
    end
end

%calculate Gram of X_Pa
X_Pa = X_Pa';
Gram_X_Pa = cal_Gram(X_Pa);

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
    if ad_matrix(K(l),i)==0 && ad_matrix(K(l),j)==0
        X_Pa_ind = K(l);
        X_Pa = cat(1,X_Pa,X{X_Pa_ind});
    end
end

%calculate Gram of X_Pa
X_Pa = X_Pa';
Gram_X_Pa = cal_Gram(X_Pa);

%calculate the estimate of R, AA, S and ASA.
Kj = Gram_X_Pa;
R = X{i}*pdinv(Kj+lam*eye(T));%m*T
AA = Kj*R'*R;
S = Kj;
ASA = Kj^2*R'*R;

%calculate difference, i.e log(tao(ASA))-log(tao(S))-log(tao(AA))
D = lnt(ASA)-lnt(S)-lnt(AA);
D2 = D;
D1
D2

%%
% % generate data from the model
% %      1   2
% %    /  \ /
% %   3 -> 4
g = pc(X,cond_ind_test,pars,maxFanIn,alpha,cache)

samplesize=1000;
alpha=0.05;
X=rand(samplesize,1)-0.5;
Y=rand(samplesize,1)-0.5;
Z=0.5*X+rand(samplesize,1)-0.5;
W=X+1.5*Z-1.9*Y+rand(samplesize,1)-0.5;
dag=pc([X Y Z W],'indtest_corr',[],2,alpha);
dag

pc(data,order,'indtest_corr',[],2,0.05)

for i= 1:100
X1=randn(300,4);
Y=X1.^2+0.5*randn(300,4);
Z=Y.^2+0.5*randn(300,4);
[p_val stat]=indtest_new(X1,Y,[],[]);
p_val % X and Z should be dependent
P(i,1)=p_val;
P(i,3)=stat;

[p_val stat]=indtest_new(X1,Z,Y,[]);
p_val % X and Z should be conditionally independent given Y
P(i,2)=p_val;
P(i,4)=stat;
end




