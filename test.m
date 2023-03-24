% normal = imread('./out/guy_normal.png');
% depth = imread('./out/guy_depth.png');
normal = imread('./out/guy_normal.png');
depth = imread('./out/guy_depth.png');

% depth = double(depth) / max(depth);
depth = double(depth);
depth = depth / max(depth(:));
normal = double(normal);
normal = normal / max(normal(:));

% imshow(normal)
% imshow(depth)
mask = zeros(size(depth, 1), size(depth, 2));
mask(1, 1) = 1;
% mask(1, :) = 1;
out = imblend(normal, mask, depth);
imwrite(out, './out/one.png');
mask(1, :) = 0;
out = imblend(normal, mask, depth);
imwrite(out, './out/row.png');
% imwrite(out, './out/guy_normal_depth2.png');
% imshow(out)
% nr = depth_to_normal(depth);
% imshow(nr)
