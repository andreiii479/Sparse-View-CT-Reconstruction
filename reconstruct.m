function [x, info] = reconstruct(prob)
    t_start = tic;
    time_limit = 115.0;

    b      = prob.b(:);
    N      = double(prob.N);
    angles = prob.angles(:);
    n_det  = double(prob.n_det);

    if isfield(prob, 'sigma') && ~isempty(prob.sigma)
        sigma = double(prob.sigma);
    else
        sigma = 0.03;
    end

    n_angles = length(angles);
    ratio    = (n_angles * n_det) / (N * N);
    if ratio >= 1.0
        lambda  = 0.02;   eps_rw = 0.02;   n_rw = 4;   tv_in = 5;
    elseif ratio >= 0.35
        lambda  = 0.04;   eps_rw = 0.03;   n_rw = 5;   tv_in = 5;
    elseif ratio >= 0.19
        lambda  = 0.25;   eps_rw = 0.10;   n_rw = 6;   tv_in = 5;
    else
        lambda  = 0.70;   eps_rw = 0.06;   n_rw = 4;   tv_in = 5;
    end
    L    = power_iter(angles, N, n_det, 20);
    L    = L * 1.01;
    step = 1.0 / L;
    if N <= 128
        iter_per_rw = 250;
    else
        iter_per_rw = 200;
    end
    x = zeros(N * N, 1);
    W = ones(N, N);
    total_iters = 0;
    rw = 0;
    px = [];
    py = [];
    for rw = 1:n_rw
        y  = x;
        t  = 1.0;
        for it = 1:iter_per_rw
            grad = adjoint_op(forward_op(y, angles, N, n_det) - b, ...
                              angles, N, n_det);
            z = y - step * grad;

            [Xnew2d, px, py] = tv_denoise_w(reshape(z, N, N), ...
                                            lambda * step, tv_in, px, py, W);
            x_new = max(Xnew2d(:), 0);

            t_new = (1 + sqrt(1 + 4 * t * t)) / 2;
            y     = x_new + ((t - 1) / t_new) * (x_new - x);
            rel_change = norm(x_new(:) - x(:)) / (norm(x(:)) + 1e-8);
            x     = x_new;
            t     = t_new;
            total_iters = total_iters + 1;
            
            if rel_change < 1.6e-5
                break;
            end
        end
        X2 = reshape(x, N, N);
        gx = [diff(X2, 1, 1); zeros(1, N)];
        gy = [diff(X2, 1, 2), zeros(N, 1)];
        gm = sqrt(gx .^ 2 + gy .^ 2);
        W  = 1.0 ./ (gm + eps_rw);
        W  = W / mean(W(:));

        if toc(t_start) > time_limit
            break;
        end
    end

    x = reshape(x, N, N);

    info.iters  = total_iters;
    info.lambda = lambda;
    info.eps_rw = eps_rw;
    info.n_rw   = rw;
    info.ratio  = ratio;
    info.time   = toc(t_start);
end

function L = power_iter(angles, N, n_det, iters)
    v = randn(N * N, 1);
    v = v / norm(v);
    L = 1;
    for i = 1:iters
        w  = adjoint_op(forward_op(v, angles, N, n_det), angles, N, n_det);
        nw = norm(w);
        if nw < 1e-30
            L = 1;
            return;
        end
        L = nw;
        v = w / nw;
    end
end
function [out, px, py] = tv_denoise_w(B, w, n_it, px, py, W)
    [m, n] = size(B);
    if isempty(px)
        px = zeros(m, n);
        py = zeros(m, n);
    end

    tau = 0.25;
    Bw  = B / w;

    for k = 1:n_it
        dx = [px(1, :); diff(px(1:m-1, :), 1, 1); -px(m-1, :)];
        dy = [py(:, 1), diff(py(:, 1:n-1), 1, 2), -py(:, n-1)];
        
        u = dx + dy - Bw;
        
        gx = [diff(u, 1, 1); zeros(1, n)];
        gy = [diff(u, 1, 2), zeros(m, 1)];

        d     = sqrt(gx .^ 2 + gy .^ 2) ./ W;
        denom = 1 + tau * d;
        px    = (px + tau * gx) ./ denom;
        py    = (py + tau * gy) ./ denom;
    end

    dx = [px(1, :); diff(px(1:m-1, :), 1, 1); -px(m-1, :)];
    dy = [py(:, 1), diff(py(:, 1:n-1), 1, 2), -py(:, n-1)];

    out = B - w * (dx + dy);
end