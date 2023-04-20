close all
eps = 0.1;
disparity_scale = 2^16 - 1;


% files = dir('./data/disparity/*.png');
% disparities = cell(1, length(files));
% for i = 1:length(files)
%     disparities{i} = imread(strcat('./data/disparity/', files(i).name));
%     disparities{i} = double(disparities{i});
%     disparities{i} = disparities{i} / disparity_scal  e;
% end

% files = dir('./data/normal/*.png');
% normals = cell(1, length(files));
% for i = 1:length(files)
%     normals{i} = imread(strcat('./data/normal/', files(i).name));
%     normals{i} = double(normals{i});
% end

% files = dir('./data/rgb/*.png');
% rgbs = cell(1, length(files));
% for i = 1:length(files)
%     rgbs{i} = imread(strcat('./data/rgb/', files(i).name));
%     rgbs{i} = double(rgbs{i});
% end

% % iterate through disparity, normal, and rgb images

% for cur_i = 1:length(disparities)
%     disparity = disparities{cur_i};
%     normal = normals{cur_i};
%     rgb = rgbs{cur_i};

%     depth = 1./(disparity + eps);
%     depth = depth ./ max(depth(:));

%     depth_normal = depth_to_normal(depth, normal);

%     [r, c, z] = size(normal); 
%     A = reshape(normal(:, :, 1:2), [r * c * 2, 1]);
%     A = cat(2, A, ones(r * c * 2, 1));
%     b = reshape(depth_normal(:, :, 1:2), [r * c * 2, 1]);
%     x = lsqr(A, b);

%     new_n = normal(:, :, :);
%     new_n = new_n * x(1, 1) + x(2, 1);


%     mask = ones(size(disparity, 1), size(disparity, 2));
%     gaussian_filter = fspecial('gaussian', [3, 3], 3);
%     depth_gauss = imfilter(depth, gaussian_filter);


%     tmp_normal = depth_to_normal(depth_gauss, 0);
%     dx = abs(tmp_normal(:, :, 1));
%     dy = abs(tmp_normal(:, :, 2));
%     eps = 0.015;
%     for i = 1:size(mask, 1)
%         for j = 1:size(mask, 2)
%             if j + 1 <= size(mask, 2) && dx(i, j) > eps && dx(i, j + 1) < eps
%                 mask(i, j + 1) = 0;
% %                 mask(i, j) = 0;
%             end
%             if j - 1 >= 1 && dx(i, j) > eps && dx(i, j - 1) < eps
%                 mask(i, j - 1) = 0;
% %                 mask(i, j) = 0;
%             end
%             if i + 1 <= size(mask, 1) && dy(i, j) > eps && dy(i + 1, j) < eps
%                 mask(i + 1, j) = 0;
% %                 mask(i, j) = 0;
%             end
%             if i - 1 >= 1 && dy(i, j) > eps && dy(i - 1, j) < eps
%                 mask(i - 1, j) = 0;
% %                 mask(i, j) = 0;
%             end
%         end
%     end

%     new_depth = imblend(new_n, mask, depth);
%     new_depth_normal = depth_to_normal(new_depth, 1);
%     depth_normal = depth_to_normal(depth, 1);

%     % save mask to data/out with r in name
%     % imwrite(mask, strcat('./data/out/mask_', num2str(cur_i), '.png'))
%     % imwrite(new_depth_normal, strcat('./data/out/ours_normal_', num2str(cur_i), '.png'))
%     % imwrite(depth_normal, strcat('./data/out/normal_', num2str(cur_i), '.png'))
%     % imwrite(new_depth, strcat('./data/out/ours_depth_', num2str(cur_i), '.png'))

%     imwrite(mask, strcat('./data/out/', num2str(cur_i), '_mask', '.png'))
%     imwrite(rescale(new_depth_normal, 0, 1), strcat('./data/out/', num2str(cur_i), '_normal_ours', '.png'))
%     imwrite(rescale(depth_normal, 0, 1), strcat('./data/out/', num2str(cur_i), '_normal', '.png'))
%     imwrite(rescale(new_depth, 0, 1), strcat('./data/out/', num2str(cur_i), '_depth_ours', '.png'))
%     imwrite(rescale(depth, 0, 1), strcat('./data/out/', num2str(cur_i), '_depth', '.png'))
%     imwrite(rescale(rgb, 0, 1), strcat('./data/out/', num2str(cur_i), '_rgb', '.png'))

% end


normal_rgb = imread('./out/point_2_view_0_domain_rgb-2_normal.png');
disparity = imread('./out/point_2_view_0_domain_rgb-2-dpt_swin2_large_384.png');
% normal_rgb = imread('./out/guy_normal.png');
% disparity = imread('./out/guy_depth.png');

% z_zero = normal(:, :) == 0;
% normal(z_zero) = 1;
% normal = double(normal);




normal = double(normal_rgb) ./ 255;
% normal = normal ./ (normal(:, :, 3));
% image_per_pixel_norm = sqrt(sum(normal.^2, 3));
% normal = normal ./ image_per_pixel_norm;

disparity = double(disparity);
% mx = 2^8 - 1;
disparity = disparity / disparity_scale;
depth = 1./(disparity + eps);
depth = depth ./ max(depth(:));
depth(depth > 1) = 1;
% normal = normal - normal(:, :, 3);
% depth = disparity;

% TODO make normal z that are 0 to 1

depth_normal = depth_to_normal(depth, 0);
% figure, imshow(abs(depth_normal(:, :, 1)))
% figure, imshow(depth_normal(:, :, 2))

% imshow(rescale(depth_normal, 0, 1))


% Find a and b such that a * normal + b = depth_normal
% [r,c] = size(disparity);
% A = reshape(disparity, [r * c, 1]);
% A = cat(2, A, ones(r * c, 1));
% b = reshape(new_depth, [r * c, 1]);
% x = lsqr(A, b);

[r, c, z] = size(normal); 
A = reshape(normal(:, :, 1:2), [r * c * 2, 1]);
A = cat(2, A, ones(r * c * 2, 1));
b = reshape(depth_normal(:, :, 1:2), [r * c * 2, 1]);
x = lsqr(A, b);

% [r, c, z] = size(normal); 
% A = reshape(normal(:, :), [r * c * z, 1]);
% A = cat(2, A, ones(r * c * z, 1));
% b = reshape(depth_normal(:, :), [r * c * z, 1]);
% x = lsqr(A, b);

new_n = normal(:, :, :);
new_n = new_n * x(1, 1) + x(2, 1);
% imshow(rescale(new_n(:, :), 0, 1))
% depth_normal_t = rescale(depth_normal(:, :, 1:2), 0, 1);
% new_n_t = rescale(new_n(:, :, 1:2), 0, 1);
% normal_t = rescale(normal(:, :, 1:2), 0, 1);
% figure, imshow(rescale(depth_normal_t(:, :), 0, 1))
% figure, imshow(rescale(new_n_t(:, :), 0, 1))
% figure, imshow(rescale(normal_t(:, :), 0, 1))

% figure, imshow(rescale(normal, 0, 1))
% figure, imshow(rescale(new_n, 0, 1))
% figure, imshow(rescale(depth_normal, 0, 1))







% mask = ones(size(disparity, 1), size(disparity, 2));
% new_depth = imblend(new_n, mask, depth);
% % new_depth = rescale(new_depth, 0, 1);
% % ref_depth = rescale(depth, 0, 1);
% [r, c] = size(new_depth); 
% A = reshape(ref_depth(:, :), [r * c, 1]);
% A = cat(2, A, ones(r * c, 1));
% b = reshape(new_depth(:, :), [r * c, 1]);
% x = lsqr(A, b);
% ref_depth = ref_depth * x(1, 1) + x(2, 1);

% figure, imshow(rescale(depth_to_normal(depth, 1), 0, 1))
% figure, imshow(rescale(depth_to_normal(ref_depth, 1), 0, 1))
% depth = ref_depth;





mask = ones(size(disparity, 1), size(disparity, 2));
% mask(1, :) = 0;
% mask(size(disparity, 1), :) = 0;
% mask(:, 1) = 0;
% mask(:, size(disparity, 2)) = 0;
gaussian_filter = fspecial('gaussian', [3, 3], 3);
depth_gauss = imfilter(depth, gaussian_filter);


% figure, imshow(rescale(depth_normal_gauss, 0, 1))
% tmp_normal = depth_to_normal(depth_gauss, 0);
% dx = abs(tmp_normal(:, :, 1));
% dy = abs(tmp_normal(:, :, 2));
% % dx = imdilate(dx, ones(3, 3));
% % dy = imdilate(dy, ones(3, 3));
% mask(dx > 0.02) = 0;
% mask(dy > 0.02) = 0;
% mask_ = imerode(mask, ones(16, 16));
% % mask = mask .* mask_;
% mask_ = mask_ + imdilate(dx > 0.02, ones(3, 3));
% mask_ = mask_ + imdilate(dy > 0.02, ones(3, 3));
% mask = mask_;

eps = 0.02;
thresh = 0.3;
normal_gauss = imfilter(normal, gaussian_filter);
normal_grey = rgb2gray(normal_gauss);



% figure, imshow(normal_grey)
% figure, imshow(normal_grad > 0.1)
depth_grad = imgradient(depth);
depth_grad = abs(depth_grad);

% initial_ = normal_grad > 0.1;
% % keep edge only if surface changes direction
% initial = imerode(initial_, ones(2, 2));
% % figure, imshow(initial)
% thresh = 0.04;
% for i = 1:size(initial, 1)
%     for j = 1:size(initial, 2)
%         if initial(i, j) == 1
%             up = i - 1;
%             down = i + 1;
%             left = j - 1;
%             right = j + 1;
%             while up > 0 && initial_(up, j) == 1
%                 up = up - 1;
%             end
%             while down < size(initial, 1) && initial(down, j) == 1
%                 down = down + 1;
%             end
%             while left > 0 && initial_(i, left) == 1
%                 left = left - 1;
%             end
%             while right < size(initial, 2) && initial_(i, right) == 1
%                 right = right + 1;
%             end
%             up = max(up, 1);
%             down = min(down, size(initial, 1));
%             left = max(left, 1);
%             right = min(right, size(initial, 2));
%             if abs(normal_grey(up, j) - normal_grey(down, j)) < thresh || abs(normal_grey(i, left) - normal_grey(i, right)) < thresh
%                 initial(i, j) = 0;
%             end
            
%         end
%     end
% end




% dx(dx < 0.1) = 0;
% imshow(rescale(dx, 0, 1))
% mask(dy > 0.2) = 0;
% set mask to zero for edges, i.e., where the gradient is large
% mask
% set mask to zero if dx > eps
% figure, imshow(depth_grad > eps)
% mask = mask .* (normal_grad < eps);
% filter normal gaussian

% mask = mask .* (depth_grad < eps);

% get rid of mask if it's not edge in normal
% mask = mask .* (normal_grad < 0.1);
% figure, imshow((normal_grad > thresh) .* (depth_grad > eps))
% mask = (normal_grad > thresh) .* (depth_grad > eps) ;
% mask = 1 - mask;
% mask = 1 - initial;
mask = ones(size(disparity, 1), size(disparity, 2));
% mask = imerode(mask, ones(3, 3));
mask = depth_grad < eps;


% numColors = 3;
% L = imsegkmeans(normal_rgb, numColors);
% B = labeloverlay(normal_rgb, L);
% figure, imshow(B)
% title("Labeled Image RGB")

% show first color only
% figure, imshow(L == 1)
% L_grad = imgradient(L);
% mask = L_grad > 0;
% mask = imdilate(mask, ones(5, 5));
% figure, imshow(mask)
% figure, imshow(depth_grad > eps)
% mask = mask .* (depth_grad > eps);



% nei_pixel = 1;
% for i = 1:size(mask, 1)
%     for j = 1:size(mask, 2)
%         if j + nei_pixel <= size(mask, 2) && dx(i, j) > eps && dx(i, j + nei_pixel) < eps
%             mask(i, j + nei_pixel) = 0;
%         end
%         if j - nei_pixel  >= 1 && dx(i, j) > eps && dx(i, j - nei_pixel) < eps
%             mask(i, j - nei_pixel) = 0;
%         end
%         if i + nei_pixel <= size(mask, 1) && dy(i, j) > eps && dy(i + nei_pixel, j) < eps
%             mask(i + nei_pixel, j) = 0;
%         end
%         if i - nei_pixel >= 1 && dy(i, j) > eps && dy(i - nei_pixel, j) < eps
%             mask(i - nei_pixel, j) = 0;
%         end
%     end
% end
% for i = 1:10
%     % set random row to zero
%     mask(randi(size(mask, 1)), randi(size(mask, 2))) = 0;
% %     mask(randi(size(mask, 1)), :) = 0;
% 
% end
% mask(200:300, 200:300) = 1;
% mask = ones(size(disparity, 1), size(disparity, 2));




% mask(1:30:end, :) = 0;

% new_depth = imblend(normal, mask, depth);
% figure, imshow(rescale(new_depth, 0, 1))


depth_grad_mask = (depth_grad > 0.2);
% normal_grad_mask = normal_grad > 0.5;
% figure, imshow(B)
% figure, imshow(normal)
depth_grad_mask = imdilate(depth_grad_mask, ones(3, 3));
% figure, imshow(depth_grad_mask)

mask = imdilate(mask, ones(3, 3));
% mask = mask | depth_grad_mask;

depth_grad = imgradient(depth);
depth_grad = abs(depth_grad) / 2;
% mask = mask + depth_grad;




% mask = imdilate(mask, ones(3, 3));
normal_grad = imgradient(normal_grey);
normal_grad = abs(normal_grad) / 2;
normal_grad(normal_grad > 1) = 1;
% normal_grad(normal_grad < 0.5) = 0;
% mask = abs(depth_grad) + (abs(normal_grad) > 0.1);
edges = abs(depth_grad);
edges(edges > 1) = 1;
combined = edges .* normal_grad;
% combined = combined ./ max(combined(:));
combined = combined > 0.1;
% edges(edges < 0.1) = 0;
% mask = edges > 0.05;

% mask = (depth_grad > 0.05);
% mask = mask + (depth_grad < 0.1 & normal_grad > 0.02) / 2;


% mask = combined + (edges > 0.1);
mask = combined;

% mask = mask + (1 - depth) / 10;
% mask = mask + (normal_grad > 0.5);
% mask = normal_grad + depth_grad;
mask = imdilate(mask, ones(8, 8));
% mask = mask ./ 5;
mask = mask ./ max(mask(:));

mask = edges + normal_grad / 5 + (edges > 0.02) / 10;
% mask = mask ./ max(mask(:));
mask(mask > 1) = 1;
mask = imdilate(mask, ones(8, 8));
mask = 1 - mask;
% mask = mask ./ 2;


mask(1:5, :) = 0;
mask(end - 5:end, :) = 0;
mask(:, 1:5) = 0;
mask(:, end - 5:end) = 0;

for i = 1:10
    % set random row to zero
    % mask(randi(size(mask, 1)), randi(size(mask, 2))) = 0.5;
    sz = 60
    rnd_i = randi(size(mask, 1) - sz);
    rnd_j = randi(size(mask, 2) - sz);
    % 10x10
    % mask(rnd_i:rnd_i + sz, rnd_j:rnd_j + sz) = 0.8;
    mask(rnd_i:rnd_i + sz, rnd_j:rnd_j + sz) = min(mask(rnd_i:rnd_i + sz, rnd_j:rnd_j + sz), 0.8);
    % mask(randi(size(mask, 1)), :) = 0.1;
    % mask(:, randi(size(mask, 2))) = 0.1;
end

figure, imshow(mask)
% mask(mask == 1) = 0.1;
depth_normal = depth_to_normal(depth, 0);
new_depth = imblend(new_n, mask, depth, depth_normal);

% fit a and b such that a * new_depth + b = depth
A = [new_depth(:), ones(size(new_depth(:)))];
b = depth(:);
x = A \ b;
fitted = x(1) * new_depth + x(2);

% fit a and b such that a * new_depth + b = depth
A = [fitted(:), ones(size(fitted(:)))];
b = depth(:);
x = A \ b;
fitted = x(1) * fitted + x(2);
% figure, imshow(rescale(depth, 0, 1))
% figure, imshow(fitted)

% figure, imshow(rescale(depth, 0, 1))

% colormap(cm_inferno)
% figure, imshow(rescale(depth, 0, 1))
% colormap(cm_inferno)
% figure, imshow(1 ./ (depth + 0.001))

new_depth_normal = depth_to_normal(new_depth, 1);
figure, imshow(rescale(new_depth_normal, 0, 1))

depth_normal = depth_to_normal(depth, 1);
figure, imshow(rescale(depth_normal, 0, 1))

cm_inferno = inferno(100);
% colormap(cm_inferno)
figure, imshow(rescale(fitted, 0, 1))
colormap(cm_inferno);

figure, imshow(rescale(depth, 0, 1))
colormap(cm_inferno);



% imwrite(out, './out/new_depth.png')
