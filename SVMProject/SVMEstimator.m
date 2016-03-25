classdef SVMEstimator < handle
    
    properties (SetAccess = public)
        C
        kernel_type
        p
        mu
        stdev
        sv_data
        sv_alpha
        sv_label
        b
    end
    
    methods
        
        function this = SVMEstimator(C, kernel_type, p)
            this.C = C;
            this.kernel_type = kernel_type;
            this.p = p;
            
        end
        
        function fit(this, X, Y)
            % Parameters
            % 	X: array-like size (n_features, n_examples)
            % 	Y: array-like size (1, n_examples)
            
            [n_features, n_train] = size(X);
            
            % normalize data to be 0 mean, unit variance
            this.mu = mean(X, 2);
            this.stdev = std(X, 0, 2);
            X = bsxfun(@rdivide, bsxfun(@minus, X, this.mu), this.stdev);
            
            if strcmp(this.kernel_type,'linear')
                h1 = Y*Y'; % d_i * d_j
                K = X'*X; % X_i * X_j
                H = h1.*K; % d_i * d_j * X_i * X_j
            else % polynomial kernel
                h1 = Y*Y'; % d_i * d_j
                K = (X'*X + 1).^this.p; % K(X_i, X_j) = Where K is Polynomial
                H = h1.*K; % d_i * d_j * K(X_i, X_j)
            end
            
            f = -1 * ones(1, n_train);
            Aeq = Y'; % labels
            Beq = 0; % alpha_i*d_i = 0
            lb = zeros(n_train, 1); % alpha lower bound = 0
            ub = this.C * ones(n_train, 1); % and upper bound = C
            alpha0 = [];
            options = optimoptions('quadprog','Algorithm','interior-point-convex');
            [alpha,~,~]=quadprog(H,f,[],[],Aeq,Beq,lb,ub,alpha0,options);
            
            thresh = max(alpha) * 0.01;
            sv_idx = find(alpha > thresh); % support vectors are the non-zeroish vectors
            
            % save support vectors
            this.sv_data = X(:,sv_idx);
            this.sv_alpha = alpha(sv_idx);
            this.sv_label = Y(sv_idx);
            
            % average b
            % b = 1 / d_i - sum(alpha_j*d_j*K(x_j, x_i))
            % where i is support vector, j is data example
            bs = 1 ./ this.sv_label' - sum(bsxfun(@times, alpha .* Y, K(:,sv_idx)),1);
            this.b = mean(bs);
            
            % choose a random b
            % rand_sv = sv_idx(randperm(length(sv_idx),1));
            % this.b = 1 / Y(rand_sv) - sum(alpha .* Y .* K(:,rand_sv));
        end
        
        function g = discriminate(this, X)
            % Calculate discriminate function. Assumes data is unnormalized
            % Parameters
            % 	X: array-like size (n_features, n_examples)
            
            % normalize test according to scaling factors of train
            X = bsxfun(@rdivide, bsxfun(@minus, X, this.mu), this.stdev);
            if strcmp(this.kernel_type,'linear')
                K = this.sv_data'*X;
            else
                K = (this.sv_data'*X + 1).^this.p;
            end
            g = (this.sv_alpha .* this.sv_label)'*K + this.b;
            g = g'; % make output (1, n_examples)
        end

        function pred = predict(this, X)
            % Calculates positive and negative predictions. Assumes data is
            % unormalized.
            % Parameters
            % 	X: array-like size (n_features, n_examples)
            g = this.discriminate(X);
            pred = sign(g);
        end
        
    end
    
end