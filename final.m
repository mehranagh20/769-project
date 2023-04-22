    close all
eps = 0.1;
disparity_scale = 2^16 - 1;


base = './new-data-out/';
dir_name = './new-data-out/';
% dir_name = './test-data-final/motorcycle';

files = dir(strcat(dir_name, '*depth_gt.png')); 
depth_gt = cell(1, length(files));
disparity_gt = cell(1, length(files));
for i = 1:length(files)
    depth_gt{i} = imread(strcat(base, files(i).name));
    depth_gt{i} = double(depth_gt{i});
    % depth_gt{i} = depth_gt{i};
end


files = dir(strcat(dir_name, '*depth.png'));
disparities = cell(1, length(files));
for i = 1:length(files)
    disparities{i} = imread(strcat(base, files(i).name));
    disparities{i} = double(disparities{i});
    disparities{i} = disparities{i} / disparity_scale;
end

files = dir(strcat(dir_name, '*normal.png'));
normals = cell(1, length(files));
for i = 1:length(files)
    normals{i} = imread(strcat(base, files(i).name));
    normals{i} = double(normals{i});
end

files = dir(strcat(dir_name, '*rgb.png'));
rgbs = cell(1, length(files));
for i = 1:length(files)
    rgbs{i} = imread(strcat(base, files(i).name));
    rgbs{i} = double(rgbs{i});
end

% % iterate through disparity, normal, and rgb images

for cur_i = 1:length(disparities)
    disparity = disparities{cur_i};
    normal = normals{cur_i};
    rgb = rgbs{cur_i};

    normal = double(normal) ./ 255;
    disparity = double(disparity);
    disparity = disparity / max(disparity(:));
    depth = 1./(disparity + eps);
    depth = depth ./ max(depth(:));
    depth(depth > 1) = 1;
    % figure, imshow(rescale(depth, 0, 1))
    % figure, imshow(rescale(normal, 0, 1))

    depth_normal = depth_to_normal(depth, 0);

    [r, c, z] = size(normal); 
    A = reshape(normal(:, :, 1:2), [r * c * 2, 1]);
    A = cat(2, A, ones(r * c * 2, 1));
    b = reshape(depth_normal(:, :, 1:2), [r * c * 2, 1]);
    x = lsqr(A, b);

    new_n = normal(:, :, :);
    new_n = new_n * x(1, 1) + x(2, 1);
    normal = new_n;

    depth_grad = imgradient(depth);
    depth_grad = abs(depth_grad);

    normal_grey = rgb2gray(normal);
    normal_grad = imgradient(normal_grey);
    normal_grad = abs(normal_grad);

    mask = ones(size(disparity, 1), size(disparity, 2));
    mask = depth_grad < eps;


    mask = depth_grad + normal_grad / 3 + (depth_grad > 0.01) / 5;
    mask(mask > 1) = 1;
    mask = imdilate(mask, ones(16, 16));
    mask = 1 - mask;

    mask(1:5, :) = 0;
    mask(end - 5:end, :) = 0;
    mask(:, 1:5) = 0;
    mask(:, end - 5:end) = 0;

    for i = 1:15
        sz = 60;
        rnd_i = randi(size(mask, 1) - sz);
        rnd_j = randi(size(mask, 2) - sz);
        mask(rnd_i:rnd_i + sz, rnd_j:rnd_j + sz) = min(mask(rnd_i:rnd_i + sz, rnd_j:rnd_j + sz), 0.8);
    end

    % figure, imshow(mask)
    depth_normal = depth_to_normal(depth, 0);
    new_depth = imblend(new_n, mask, depth, depth_normal);

    % new_depth = rescale(new_depth, 0, 1);
    % depth = rescale(depth, 0, 1);

    A = [new_depth(:), ones(size(new_depth(:)))];
    b = depth(:);
    x = A \ b;
    fitted = x(1) * new_depth + x(2);

    new_depth_normal = depth_to_normal(fitted, 1);
    % figure, imshow(rescale(new_depth_normal, 0, 1))

    depth_normal = depth_to_normal(depth, 1);
    % figure, imshow(rescale(depth_normal, 0, 1))

    cm_inferno = inferno(100);


    fitted = fitted ./ max(fitted(:));
    depth = depth ./ max(depth(:));
    imwrite(mask, strcat('./data/out/', files(cur_i).name, '_mask', '.png'))
    imwrite(rescale(new_depth_normal, 0, 1), strcat('./data/out/',files(cur_i).name, '_normal_ours', '.png'))
    imwrite(rescale(depth_normal, 0, 1), strcat('./data/out/', files(cur_i).name, '_normal', '.png'))
    imwrite(rescale(rgb, 0, 1), strcat('./data/out/', files(cur_i).name, '_rgb', '.png'))
    imwrite(ind2rgb(uint8(fitted * 255), cm_inferno), strcat('./data/out/', files(cur_i).name, '_depth_ours', '.png'))
    imwrite(ind2rgb(uint8(depth * 255), cm_inferno), strcat('./data/out/', files(cur_i).name, '_depth', '.png'))
    imwrite(rescale(fitted, 0, 1), strcat('./data/out/', files(cur_i).name, '_depth_grey_ours', '.png'))
    imwrite(rescale(depth, 0, 1), strcat('./data/out/', files(cur_i).name, '_depth_grey', '.png'))

    % fit a and b for new_depth and depth_gt
    A = [new_depth(:), ones(size(new_depth(:)))];
    b = depth_gt{cur_i}(:);
    x = A \ b;
%     new_depth = x(1) * new_depth + x(2);
    new_depth_dis = 1 ./ (new_depth + 0.001);
    % figure, imshow(rescale(new_depth_dis, 0, 1))

    depth_dis = 1 ./ (depth + 0.001);
    
    cur_depth_gt = depth_gt{cur_i};
    cur_depth_gt = rescale(cur_depth_gt, 0, 1);
    gt_dis = 1 ./ (cur_depth_gt + 0.05);
    % figure, imshow(rescale(gt_dis, 0, 1))

    imwrite(rescale(new_depth_dis, 0, 1), strcat('./data/out/', files(cur_i).name, '_disparity_ours', '.png'))
    imwrite(rescale(depth_dis, 0, 1), strcat('./data/out/', files(cur_i).name, '_disparity', '.png'))
    imwrite(rescale(gt_dis, 0, 1), strcat('./data/out/', files(cur_i).name, '_disparity_gt', '.png'))

end

