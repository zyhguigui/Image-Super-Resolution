function [B, S, stat] = reg_sparse_coding(X, num_bases, Sigma, beta, gamma, num_iters, batch_size, initB, fname_save)
%
% Regularized sparse coding, train dictionaries
%
% Inputs
%       X           -data samples, column wise
%       num_bases   -number of bases
%       Sigma       -smoothing matrix for regularization
%       beta        -smoothing regularization
%       gamma       -sparsity regularization
%       num_iters   -number of iterations 
%       batch_size  -batch size
%       initB       -initial dictionary
%       fname_save  -file name to save dictionary
%
% Outputs
%       B           -learned dictionary
%       S           -sparse codes
%       stat        -statistics about the training
%
% Written by Jianchao Yang @ IFP UIUC, Sep. 2009.
%
% Additional notes on algorithm:
%    1. Initialize dictionary B;
%    2. Fix B, train sparse codes S;
%    3. Fix S, train dictionary B;
%    4. Iterate between 2 and 3 for a fixed number of times (set by num_iters).
%
% Written on 2017/01/03

pars = struct;
pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_trials = num_iters;
pars.beta = beta;
pars.gamma = gamma;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom

% original
% if ~isa(X, 'double')
%     X = cast(X, 'double');
% end

% Modified 2017/01/05
if ~isa(X, 'single')
    X = cast(X, 'single');
end

if isempty(Sigma)
	Sigma = single(eye(pars.num_bases));
end

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X, 2);
end

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('Temporary dictionary\\reg_sc_b%d_%s', num_bases, datestr(now, 30));	
end

pars %#ok<NOPRT>

%% initialize basis/Dictionary
if ~exist('initB','var') || isempty(initB)
    B = single(rand(pars.patch_size, pars.num_bases)-0.5); % Uniformly distributed random numbers in the interval (-0.5,0.5)
	B = B - repmat(mean(B,1), size(B,1),1);
    B = B * diag(1./sqrt(sum(B.*B)));
else
    disp('Using initial B...');
    B = initB;
end

% [L, M]=size(B); % Unused variables, 2017/01/02

% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;

%% Train dictionaries
% t=0;                       % original codes
% while t < pars.num_trials
%     t = t + 1;
for t = 1:pars.num_trials    % Modified 2017/01/03
    % start_time = cputime;  % original code
    tic;        % Modified 2017/01/03
    stat.fobj_total = 0;    
    % Take a random permutation of the samples
    indperm = uint32(randperm(size(X,2)));
    
    sparsity = [];
    
    for batch=1:(size(X,2)/pars.batch_size)
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx); % Xb: a batch of X
        
        % original code
        % learn coefficients (conjugate gradient)   
        % S = L1QP_FeatureSign_Set(Xb, B, Sigma, pars.beta, pars.gamma); % solved through linear
        % programming 
        
        % ============== replace L1QP_FeatureSign_Set() ================= %
        [~, nSmp] = size(Xb);
        nBases = size(B, 2);
        
        % sparse codes of the features
        % S = sparse(nBases, nSmp); % original code
        S = single(zeros(nBases, nSmp));    % modified 2017/01/01
        
        A = B'*B + 2 * pars.beta * Sigma;
        
        for ii = 1:nSmp
            % b = -B' * Xb(:, ii);  % original code, the slowest line!!!!!!!
            b = -(Xb(:, ii)' * B)'; % modified 2017/01/07
            %     [net] = L1QP_FeatureSign(gamma, A, b);
            S(:, ii) = L1QP_FeatureSign_yang(gamma, A, b);
        end
        % ============== replace L1QP_FeatureSign_Set() ================= %
   
        sparsity(end+1) = length(find(S(:) ~= 0))/length(S(:));
        
        % get objective
        % [fobj] = getObjective_RegSc(Xb, B, S, Sigma, pars.beta, pars.gamma); % original code
        
        % ============== replace getObjective_RegSc() ================= %
        Err = Xb - B*S;
        fresidue = 0.5*sum(sum(Err.^2));
        fsparsity = gamma*sum(sum(abs(S)));
        
        fregs = 0;
        for ii = 1:size(S, 1)  % oroginal: for ii = size(S,1) ???
            fregs = fregs + beta*S(:, ii)'*Sigma*S(:, ii);
        end
        
        fobj = fresidue + fsparsity + fregs;
        % ============== replace getObjective_RegSc() ================= %
        
        stat.fobj_total = stat.fobj_total + fobj;
        
        % update basis
        B = l2ls_learn_basis_dual(Xb, S, pars.VAR_basis); % Quadratically constrained quadratic programming 
    end
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    % stat.elapsed_time(t)  = cputime - start_time; % original code
    stat.elapsed_time(t) = toc;  % modified 2017/01/03
    
    fprintf(['epoch= %d/%d, sparsity = %f, fobj= %f, took %0.2f ' ...
             'seconds\n'], t, pars.num_trials, mean(sparsity), stat.fobj_avg(t), stat.elapsed_time(t));
         
    % save results, create directory if it does not exist
    fprintf('saving results ...\n');
    experiment = [];
    experiment.matfname = sprintf('%s.mat', pars.filename);  
    try
        save(experiment.matfname, 't', 'pars', 'B', 'stat');
    catch
        dir = strsplit(experiment.matfname,'\');
        mkdir(dir{1:end-1});
        save(experiment.matfname, 't', 'pars', 'B', 'stat');
    end
    fprintf('Temporary dictionaries saved in %s\n', experiment.matfname);
end

end

% function retval = assert(expr) % Unused funtion, 2017/01/03
% retval = true;
% if ~expr 
%     error('Assertion failed');
%     retval = false;
% end
% return
