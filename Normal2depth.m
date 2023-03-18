function [depth] = Normal2depth(Normal)

    r=size(Normal,1);
    c=size(Normal,2);
    depth=zeros(r,c);
    depth(1,1)=0;
    for i=2:c
    depth(i,1)=depth(i-1,1)+Normal(i-1,1,1);
    end
    for i=2:r
        depth(:,i)=depth(:,i-1)+Normal(:,i-1,2);
    end
end