function  output = imblend(source3, mask3, target3)
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
                if mask(row, col) == 0
                    ind = ind + 1;
                    sparse_i(ind) = flat;
                    sparse_j(ind) = flat;
                    sparse_s(ind) = 1;
                    b(flat) = target(row, col);
                else 
                    b_val = 0;
                    % n_is = [row - 1, row + 1, row, row];
                    % n_js = [col, col, col - 1, col + 1];
                    n_is = [row - 1, row];
                    n_js = [col, col - 1];
                    num_nei = 0;
                    for n_ind = 1: 2
                        n_i = n_is(n_ind);
                        n_j = n_js(n_ind);
                        if n_i >= 1 && n_i <= size(source, 1) && n_j >= 1 && n_j <= size(source, 2)
                            num_nei = num_nei + 1;
                            if mask(n_i, n_j) == 1
                                ind = ind + 1;
                                sparse_i(ind) = flat;
                                sparse_j(ind) = (n_i - 1) * size(source, 2) + n_j;
                                sparse_s(ind) = -1;
                                if n_i == row - 1
                                    b_val = b_val + source(row, col, 0);
                                else
                                    b_val = b_val + source(row, col, 1);
                                end
                            else 
                                if n_i == row - 1
                                    b_val = b_val + source(row, col, 0) + target(n_i, n_j);
                                else
                                    b_val = b_val + source(row, col, 1) + target(n_i, n_j);
                                end
                                % b_val = b_val + source(row, col) - source(n_i, n_j) + target(n_i, n_j); 
                            end
                        end
                    end
                    ind = ind + 1;
                    sparse_i(ind) = flat;
                    sparse_j(ind) = flat;
                    sparse_s(ind) = num_nei;
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
    
    