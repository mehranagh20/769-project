function  image = depth_to_normal(depth)
    image = zeros(size(depth, 1), size(depth, 2), 3);
    z = 1;
    dx = [diff(depth, 1, 2), depth(:, 1) - depth(:, end)];
    dy = [diff(depth, 1, 1); depth(1, :) - depth(end, :)];
    grad = cat(3, dx, dy, z * ones(size(depth)));
    norm = sqrt(sum(grad.^2, 3));
    image(:, :, 1) = -dx ./ norm;
    image(:, :, 2) = -dy ./ norm;
    image(:, :, 3) = z * ones(size(depth)) ./ norm;
    
    