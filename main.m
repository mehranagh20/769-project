guy_normal=imread("./guy_normal.png");
guy_depth=imread("./guy_depth.png");
guyd=double(rgb2gray(guy_depth));
[r,c]=size(guyd);
Guy=double(guy_normal);
Guy=Guy./sum((255-Guy).^2, 3);
out=Normal2depth(Guy,(1./guyd).*(min(min(guyd))));
out1=out-min(min(out));
out2=out1/max(max(out1))*255;
imshow(uint8(round(out2)));
%Hout=adapthisteq(uint8(round(out2)));
%imshow(Hout);
figure(2)
imshow(uint8(guyd));
A=reshape(guyd,[r*c,1]);
A=cat(2,A,ones(r*c,1));
b=reshape(out2,[r*c,1]);
x = lsqr(A,b);
out3=(out2-x(2,1))./x(1,1);
figure(3)
imshow(uint8(round(out3)));
%figure(4)
%Hout=adapthisteq(uint8(round(out3)));
%imshow(Hout);
