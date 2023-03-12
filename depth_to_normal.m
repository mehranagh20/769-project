% matlab implementation of depth map to normal map
function  output = depth_to_normal( depth )
    z = 0.001
    dx = depth(:, [2:end end]) - depth;
    dy = depth([2:end end], :) - depth;
    output = zeros(size(depth,1), size(depth,2), 3);
    grad_appended_one = cat(3, -dx, -dy, ones(size(depth)) .* z);
    grad_norm = sum(grad_appended_one.^2, 3);
    nx = -dx ./ grad_norm;
    ny = -dy ./ grad_norm;
    nz = ones(size(depth)) .* z ./ grad_norm;
    output(:,:,1) = nx;
    output(:,:,2) = ny;
    output(:,:,3) = nz;
    % normalized_out = output ./ sqrt(sum(output.^2, 3));
    % output = normalized_out;
    % output(:, :, 3) = 0.001;
    % normalize each pixel to unit length
    % output = output ./ sqrt(sum(output.^2, 3));
    % output(:, :, :) = output(:, :, :) * 100;
end