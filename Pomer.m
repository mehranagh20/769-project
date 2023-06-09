function  output = Pomer(source, mask, target, transparent )

source_x=source(:,:,1);
source_y=source(:,:,2);
source_z=source(:,:,3);
%source_x=source_x/source_z;
%source_y=source_y/source_z;



if nargin<4
    transparent=0;
end

if size(mask,3)>1
    mask=rgb2gray(mask);
end
target=double(target);
source=double(source);
output=target;
laplacian_x=[1 -1];
laplacian_y=[1;-1];
mask=mat2gray(mask);
mask=logical(mask);
n=size(find(mask==1),1);
map=zeros(size(mask));
counter=0;
for x=1:size(map,1)
    for y=1:size(map,2)
        if mask(x,y)==1 %is it unknow pixel?
            counter=counter+1;
            map(x,y)=counter;  %map from (x,y) to the corresponding pixel
            %in the 1D vector
        end
    end
end
    
    coeff_num=5;
    
    A=spalloc(n,n,n*coeff_num);
    B=zeros(n,1);
 if transparent==1  % mixing gradients
%         %create the gradient mask for the first derivative
        %grad_mask_x=[-1 1];
        %grad_mask_y=[-1;1]; 
        
        get the first derivative of the target image
        g_x_target=conv2(target,grad_mask_x, 'same');
        g_y_target=conv2(target,grad_mask_y, 'same');
        g_mag_target=sqrt(g_x_target.^2+g_y_target.^2);
        
        %get the first derivative of the source image
        g_x_source=source_x;
        g_y_source=source_y;
        g_mag_source=sqrt(g_x_source.^2+g_y_source.^2);
        
        %work with 1-D
        g_mag_target=g_mag_target(:);
        g_mag_source=g_mag_source(:);
        
        %initialize the final gradient with the source gradient
        g_x_final=g_x_source(:);
        g_y_final=g_y_source(:);
        
        %if the gradient of the target image is larger than the gradient of
        %the source image, use the target's gradient instead
        g_x_final(abs(g_mag_target)>abs(g_mag_source))=...
            g_x_target(g_mag_target>g_mag_source);
        g_y_final(abs(g_mag_target)>abs(g_mag_source))=...
            g_y_target(g_mag_target>g_mag_source);
        
        %map to 2-D
        g_x_final=reshape(g_x_final,size(source,1),size(source,2));
        g_y_final=reshape(g_y_final,size(source,1),size(source,2));
        
        %get the final laplacian of the combination between the source and
        %target images lap=second deriv of x + second deriv of y
        lap=conv2(g_x_final,grad_mask_x, 'same');
        lap=lap+conv2(g_y_final,grad_mask_y, 'same');
%         
else
    
        lap=conv2(source_x,laplacian_x, 'same')+conv2(source_y,laplacian_y, 'same');
end
    %end
    counter=0;
    for x=1:size(map,1)
        for y=1:size(map,2)
            if mask(x,y)==1
                counter=counter+1;
                A(counter,counter)=4;
                
                if x>1
                if  mask(x-1,y)==0
                    B(counter,1)=target(x-1,y); 
                else 
                    A(counter,map(x-1,y))=-1; 
                end
                end
                if x<size(map,1)
                if  mask(x+1,y)==0 
                    B(counter,1)=B(counter,1)+target(x+1,y); 
                else 
                    A(counter,map(x+1,y))=-1; 
                end
                end
                if y>1 
                    if mask(x,y-1)==0 
                    B(counter,1)=B(counter,1)+target(x,y-1); 
                    else
                    A(counter,map(x,y-1))=-1; 
                    end
                end
                if y<size(map,2)
                    if mask(x,y+1)==0 
                    B(counter,1)=B(counter,1)+target(x,y+1); 
                    else 
                    A(counter,map(x,y+1))=-1; 
                    end
                end
                                
                B(counter,1)=B(counter,1)-lap(x,y);
                
            end
        end
    end
    
    X=A\reshape(B,[n,1]);
    for counter=1:length(X)
        [index_x,index_y]=find(map==counter);
        output(index_x,index_y)=X(counter);
        
    end
end



