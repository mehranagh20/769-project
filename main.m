guy_normal=imread("./guy_normal.png");
Guy=double(guy_normal);
Guy=Guy./(Guy(:,:,1).^2+Guy(:,:,2).^2+Guy(:,:,3).^2);
out=Normal2depth(Guy);
out=out/max(max(out))*255;
imshow(uint8(round(out)));
Hout=adapthisteq(uint8(round(out)));
imshow(Hout);
