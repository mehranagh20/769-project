
function []= make_3_d (rgb,depth,new_depth, parameterford,parameterfornewd)

focalLength      = [53.54, 53.92];
principalPoint   = [192, 192];
imageSize        = size(depth,[1,2]);
intrinsics       = cameraIntrinsics(focalLength,principalPoint,imageSize);
depthScaleFactor = 5e1;
maxCameraDepth   = 5;
depth_converted=uint8(rescale(new_depth, parameterford, 255));
depth_orginal=uint8(rescale(depth, parameterfornewd, 255));
ptCloud = pcfromdepth(depth_orginal,depthScaleFactor, intrinsics, ...
                      ColorImage=rgb, ...
                      DepthRange=[parameterford maxCameraDepth]);
figure();pcshow(ptCloud, VerticalAxis="Y", VerticalAxisDir="Up", ViewPlane="YX");

ptCloud = pcfromdepth(depth_converted,depthScaleFactor, intrinsics, ...
                      ColorImage=rgb, ...
                      DepthRange=[parameterfornewd maxCameraDepth]);
figure();pcshow(ptCloud, VerticalAxis="Y", VerticalAxisDir="Up", ViewPlane="YX");

end
