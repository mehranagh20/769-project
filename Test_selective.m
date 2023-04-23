eps = 0.1;
disparity_scale = 2^16 - 1;


% normal = imread('./out/point_2_view_0_domain_rgb-2_normal.png');
% disparity = imread('./out/point_2_view_0_domain_rgb-2-dpt_swin2_large_384.png');
normal = imread('./point_2_view_0_domain_rgb-2_normal.png');
disparity = imread('./point_2_view_0_domain_rgb-2-dpt_swin2_large_384.png');
rgb=imread('./main_image.jpg');
rgb=imresize(rgb,384/512);
rgb_image=double(rgb);
main_image_gray=rgb2gray(rgb);
x=input("how many regions do you want to choose?");
imshow(rgb)

% Rectangle position is given as [xmin, ymin, width, height]
img_cropped_mask=zeros(size(rgb,1),size(rgb,2));
for i=1:x
h_rect = imrect();
pos_rect = h_rect.getPosition();
% Round off so the coordinates can be used as indices
pos_rect = round(pos_rect);
% Select part of the image
img_cropped_mask(pos_rect(2) + (0:pos_rect(4)), pos_rect(1) + (0:pos_rect(3))) =1;
end

%%

% z_zero = normal(:, :) == 0;
% normal(z_zero) = 1;
normal = double(normal);
% normal = normal ./ normal(:, :, 3);
% image_per_pixel_norm = sqrt(sum(normal.^2, 3));
% normal = normal ./ image_per_pixel_norm;

disparity = double(disparity);
% mx = 2^8 - 1;
disparity = disparity / disparity_scale;
depth = 1./(disparity + eps);
depth = depth ./ max(depth(:));
% normal = normal - normal(:, :, 3);
% depth = disparity;

% TODO make normal z that are 0 to 1

depth_normal = depth_to_normal(depth, normal);
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
x = lsqr(A, b,1e-10,100);

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


mask = ones(size(disparity, 1), size(disparity, 2));
% mask(1, :) = 0;
% mask(size(disparity, 1), :) = 0;
% mask(:, 1) = 0;
% mask(:, size(disparity, 2)) = 0;
gaussian_filter = fspecial('gaussian', [3, 3], 3);
depth_gauss = imfilter(depth, gaussian_filter);


% figure, imshow(rescale(depth_normal_gauss, 0, 1))
tmp_normal = depth_to_normal(depth_gauss, 0);
dx = abs(tmp_normal(:, :, 1));
dy = abs(tmp_normal(:, :, 2));
% dx(dx < 0.1) = 0;
% imshow(rescale(dx, 0, 1))
% mask(dy > 0.2) = 0;
% set mask to zero for edges, i.e., where the gradient is large
% mask
% set mask to zero if dx > eps

eps = 0.01;
for i = 1:size(mask, 1)
    for j = 1:size(mask, 2)
        if j + 1 <= size(mask, 2) && dx(i, j) > eps && dx(i, j + 1) < eps
            mask(i, j + 1) = 0;
            mask(i, j) = 0;
        end
        if j - 1 >= 1 && dx(i, j) > eps && dx(i, j - 1) < eps
            mask(i, j - 1) = 0;
            mask(i, j) = 0;
        end
        if i + 1 <= size(mask, 1) && dy(i, j) > eps && dy(i + 1, j) < eps
            mask(i + 1, j) = 0;
            mask(i, j) = 0;
        end
        if i - 1 >= 1 && dy(i, j) > eps && dy(i - 1, j) < eps
            mask(i - 1, j) = 0;
            mask(i, j) = 0;
        end
    end
end
%% Edge detection algorithms

Edge=edge(depth_gauss,'canny',0.4);
figure();imshow(Edge);
modified_edge=img_cropped_mask.*double(Edge);
figure();imshow(modified_edge);
%%
image_dialated=imdilate(modified_edge,ones(5,5));
figure();imshow(image_dialated-Edge);
Dif=image_dialated-Edge;

% mask(1:30:end, :) = 0;
mask=1-Dif;
figure();imshow(mask);
%mask(1:3,:)=0;
%mask(end-3:end,:)=0;
%mask(:,1:3)=0;
%mask(:,end-3:end)=0;

% new_depth = imblend(normal, mask, depth);
% figure, imshow(rescale(new_depth, 0, 1))
%%
parula=fake_parula();
magma=magma();
inferno=inferno();
plasma=plasma();
viridis=viridis();

figure, imshow(mask)

new_depth = imblend(new_n, mask, depth);
figure, imshow(rescale(new_depth, 0, 1))
%colormap(inferno)
figure, imshow(rescale(depth, 0, 1))
%colormap(inferno)
Differentce=rescale(new_depth,0,1)-rescale(depth,0,1);
figure();imshow(rescale(Differentce,0,1));

new_depth_normal = depth_to_normal(new_depth, 1);
figure, imshow(rescale(new_depth_normal, 0, 1))

depth_normal = depth_to_normal(depth, 1);
figure, imshow(rescale(depth_normal, 0, 1))


% imwrite(out, './out/new_depth.png')
%%
rgb=imread('./main_image.jpg');
rgb=imresize(rgb,384/512);
focalLength      = [53.54, 53.92];
principalPoint   = [192, 192];
imageSize        = size(depth,[1,2]);
intrinsics       = cameraIntrinsics(focalLength,principalPoint,imageSize);
depthScaleFactor = 5e1;
maxCameraDepth   = 5;
depth_converted=uint8(rescale(new_depth, 0, 255));
depth_orginal=uint8(rescale(depth, 0, 255));
ptCloud = pcfromdepth(depth_orginal,depthScaleFactor, intrinsics, ...
                      ColorImage=rgb, ...
                      DepthRange=[0 maxCameraDepth]);
figure();pcshow(ptCloud, VerticalAxis="Y", VerticalAxisDir="Up", ViewPlane="YX");

ptCloud = pcfromdepth(depth_converted,depthScaleFactor, intrinsics, ...
                      ColorImage=rgb, ...
                      DepthRange=[0 maxCameraDepth]);
figure();pcshow(ptCloud, VerticalAxis="Y", VerticalAxisDir="Up", ViewPlane="YX");
