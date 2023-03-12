normal = imread('./out/guy_normal.png');
depth = imread('./out/guy_depth.png');

depth = double(depth) / 255.0;
depth_avg = mean(depth, 3);

depth_conv = depth_to_normal(depth_avg);
imshow(depth_conv)