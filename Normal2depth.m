function [depth] = Normal2depth(Normal,D)

    r=size(Normal,1);
    c=size(Normal,2);
    depth=zeros(r,c);
    depth(1,1)=0;
    for i=2:c
    depth(i,1)=depth(i-1,1)+Normal(i,1,1);
    end
    for i=2:r
    depth(1,i)=depth(1,i-1)+Normal(1,i,2);
    end
    for i=2:r
        for j=2:c
            depth(i,j)=(depth(i,j-1)+Normal(i,j,2)+depth(i-1,j)+Normal(i,j,1)+D(i,j))/3;
        end
    end
end
