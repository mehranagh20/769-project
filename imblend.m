function  output = imblend(source3, mask3, target3, grad)
    image = zeros(size(source3, 1), size(source3, 2), 1);
    for channel = 1: 1
        ind = 0;
        sparse_i = zeros(1, 5 * size(source3, 1) * size(source3, 2));
        sparse_j = zeros(1, 5 * size(source3, 1) * size(source3, 2));
        sparse_s = zeros(1, 5 * size(source3, 1) * size(source3, 2));
        b = zeros(size(source3, 1) * size(source3, 2), 1);
        source = source3(:, :, channel);
        mask = mask3(:, :);
        target = target3(:, :);
        for row = 1: size(source, 1)
            for col = 1: size(source, 2)
                flat = (row - 1) * size(source, 2) + col;
                if mask(row, col) < 1e-2
                    ind = ind + 1;
                    sparse_i(ind) = flat;
                    sparse_j(ind) = flat;
                    sparse_s(ind) = 1;
                    b(flat) = target(row, col);
                else 
                    b_val = 0;
                    b_val_sec = 0;
                    % n_is = [row - 1, row + 1, row, row];
                    % n_js = [col, col, col - 1, col + 1];
                    n_is = [row - 1, row, row + 1, row];
                    n_js = [col, col - 1, col, col + 1];
                    num_nei = 0;
                    for n_ind = 1: 4
                        n_i = n_is(n_ind);
                        n_j = n_js(n_ind);
                        if n_i >= 1 && n_i <= size(source, 1) && n_j >= 1 && n_j <= size(source, 2)
                            cur_s = +source3(row, col, 1);
                            cur_s_sec = +grad(row, col, 1);
                            cur_s = mask(row, col) * cur_s + (1 - mask(row, col)) * cur_s_sec; 
                            if n_i == row - 1
                                cur_s = +source3(row, col, 2);
                                cur_s_sec = +grad(row, col, 2);
                                cur_s = mask(row, col) * cur_s + (1 - mask(row, col)) * cur_s_sec;

                            elseif n_i == row + 1
                                cur_s = -source3(row + 1, col, 2);
                                cur_s_sec = -grad(row + 1, col, 2);
                                cur_s = mask(row + 1, col) * cur_s + (1 - mask(row + 1, col)) * cur_s_sec;
                            elseif n_j == col + 1
                                cur_s = -source3(row, col + 1, 1);
                                cur_s_sec = -grad(row, col + 1, 1);
                                cur_s = mask(row, col + 1) * cur_s + (1 - mask(row, col + 1)) * cur_s_sec;
                            end

                            num_nei = num_nei + 1;
                            if mask(n_i, n_j) > 1e-2
                                ind = ind + 1;
                                sparse_i(ind) = flat;
                                sparse_j(ind) = (n_i - 1) * size(source, 2) + n_j;
                                sparse_s(ind) = -1;
                                b_val = b_val - cur_s;
                                b_val_sec = b_val_sec - cur_s_sec;
                            else 
                                b_val = b_val - cur_s + target(n_i, n_j); 
                                b_val_sec = b_val_sec - cur_s_sec + target(n_i, n_j);
                            end
                        end
                    end
                    ind = ind + 1;
                    sparse_i(ind) = flat;
                    sparse_j(ind) = flat;
                    sparse_s(ind) = num_nei;
                    % b(flat) = b_val * mask(row, col) + b_val_sec * (1 - mask(row, col));
                    b(flat) = b_val;
                end
            end
        end
    
        sparse_i = sparse_i(1:ind);
        sparse_j = sparse_j(1:ind);
        sparse_s = sparse_s(1:ind);
        A_sparse = sparse(sparse_i, sparse_j, sparse_s, size(source, 1) * size(source, 2), size(source, 1) * size(source, 2));
        x = A_sparse \ b;
    
        for row = 1: size(source, 1)
            for col = 1: size(source, 2)
                flat = (row - 1) * size(source, 2) + col;
                image(row, col, channel) = x(flat);
            end
        end
    end
    % output = source3 .* mask3 + target3 .* ~mask3;
    output = image;
    
    