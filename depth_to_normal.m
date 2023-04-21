function  image = depth_to_normal(depth, normalize)
    image = zeros(size(depth, 1), size(depth, 2), 3);
    z = 0.002;
    dx = [diff(depth, 1, 2), depth(:, 1) - depth(:, end)];
    dy = [diff(depth, 1, 1); depth(1, :) - depth(end, :)];
    % grad = cat(3, dx, dy, z * ones(size(depth)));
    % norm = sqrt(sum(grad.^2, 3));
    image(:, :, 1) = -dx;
    image(:, :, 2) = -dy;
    image(:, :, 3) = z;
    if normalize == 1
        image_per_pixel_norm = sqrt(sum(image.^2, 3));
        image = image ./ image_per_pixel_norm;
    end


    % python code
    % sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)
    % sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)
    % normal = np.dstack([-sx, sy, np.ones_like(sx)])
    % n = np.linalg.norm(normal, axis=2, keepdims=True)
    % normal = normal / n
    % normal = (normal + 1) / 2
    % return normal

    % matlab code

