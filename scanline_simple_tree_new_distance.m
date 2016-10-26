%%
%we can use the view1 as the reference frame, then find the disparity map
%for each pixel, which minimizes the energy function. The energy function
%is made of 2 parts, similarity + smoothness

%%
%initialization of parameters
maxd = 18;
beta1 = 1.;
beta2 = 3.;
ratio = 0.0001;
rescale = 5;
global_shift = 150;

beta1 = beta1*ratio;
beta2 = beta2*ratio;

%read in the original images, view1 is seen from 
%the left and view2 is seen from the right 
view1 = im2double(imread('C:\Users\Bing\Desktop\Graphical Model\stereo image segmentation\Art-2views\Art\view1.png'));
view2 = im2double(imread('C:\Users\Bing\Desktop\Graphical Model\stereo image segmentation\Art-2views\Art\view5.png'));
ground_truth1 = imread('C:\Users\Bing\Desktop\Graphical Model\stereo image segmentation\Art-2views\Art\disp1.png');
ground_truth2 = imread('C:\Users\Bing\Desktop\Graphical Model\stereo image segmentation\Art-2views\Art\disp5.png');
[x, y, z] = size(view1);    % x = 1110, y = 1390, view1 and view2 have the same sizes

% Now we have got the optimal global shift, we are ready to find the
% disparity matrix, the naive way is like the denoysing, update the whoe
% network until convergence of all nodes.
view1 = view1(:, (global_shift+1):end, :);
view2 = view2(:, 1:(y-global_shift), :);
y = y - global_shift;
x = x/rescale;
y = y/rescale;
sview1 = zeros(x,y,3);
sview2 = zeros(x,y,3);

for i = 1:x
    for j = 1:y
        square = view1((rescale*i-rescale+1):(rescale*i),(rescale*j-rescale+1):(rescale*j),:);
        tmp = reshape(sum(sum(square, 1),2),3,1);
        sview1(i, j, :) = tmp;
        square =  view2((rescale*i-rescale+1):(rescale*i),(rescale*j-rescale+1):(rescale*j),:);
        tmp = reshape(sum(sum(square, 1),2),3,1);
        sview2(i, j, :) = tmp;
    end
end
view1=sview1/rescale^2;
view2=sview2/rescale^2;
%Now we have completed the cut and preparation 

%Now let's compute the Vertical Tree, and get optimal energy matrix V(p, dp)
%First find C(p,dp) on each column, i.e. only maximize the energy on a
%line. C can be calculated by F(forward) and B(backward) matrix
Fv = ones(x, y, 2*maxd+1) * inf;
Bv = ones(x, y, 2*maxd+1) * inf;
mv = ones(x, y, 2*maxd+1) * inf; 
%mv and mh are actually the same, both stand for the color distance between
%p in view1 and p+dj in view2, not dependent on the order and direction we 
%are scanning the picture.

fprintf('calculating scanlines...\n');
for col = 2:(y-1)
    dt = ones(x, 2*maxd+1) * inf;
    
    %initialize the first line of the dynamic table, the leftmost pixel
    for j = 1:(2*maxd+1)
        dp = j-maxd-1;
        if (col + dp<2) || (col+dp >y-1)
            continue;
        end
        %fprintf('adfa\n');
        cvec1 = reshape(permute(view1(1:3, (col-1):(col+1),:),[2,1,3]), 9, 3);
        cvec2 = reshape(permute(view2(1:3, (col+dp-1):(col+dp+1),:),[2,1,3]), 9, 3);
        dt(2, j) = colordistance_3elem(cvec1,cvec2);
        mv(2, col, j) = dt(1, j);
    end
    
    %now start to update the dynamic table
    for i = 3:(x-1)
        for j = 1:(2*maxd+1)
            dq = j - maxd - 1;
            if col+dq<2 || col+dq>y-1
                continue;
            end
            cvec1 = reshape(permute(view1((i-1):(i+1), (col-1):(col+1),:),[2,1,3]), 9, 3);
            cvec2 = reshape(permute(view2((i-1):(i+1), (col+dq-1):(col+dq+1),:),[2,1,3]), 9, 3);
            dt(i, j) = colordistance_3elem(cvec1,cvec2);
            dist = dt(i, j);
            mv(i, col, j) = dist;
            for k = 1:(2*maxd+1)
                dp = k - maxd -1;
                dqp = abs(dq - dp);
                if dt(i-1, k) < Inf
                    discont = 0;
                    if dqp==1
                        discont = beta1;
                    else if dqp>1
                            discont = beta2;
                        end
                    end
                    tmp = dt(i-1, k) + discont; 
                    if dt(i, j) > tmp
                        dt(i, j) = tmp;
                    end
                end
            end
            
            dt(i,j) = dt(i,j)+dist;
        end
    end
    Fv(:,col,:) = dt;
    %fprintf('%d %f\n', col, Fv(1, col, maxd+1));
end

for col = 2:(y-1)
    dt = ones(x, 2*maxd+1) * inf;
    line1 = reshape(view1(:,col,:),x,3);
    %initialize the first line of the dynamic table, the leftmost pixel    
    dt(x-1, :) = mv(x-1, col, :);
    
    %now start to update the dynamic table
    for i = (x-2):-1:2
        for j = 1:(2*maxd+1)
            dq = j - maxd - 1;
            if col+dq<1 || col+dq>y
                continue;
            end
            dist = mv(i, col, j);
            
            for k = 1:(2*maxd+1)
                dp = k - maxd -1;
                dqp = abs(dq - dp);
                if dt(i+1, k) < Inf
                    discont = 0;
                    if dqp==1
                        discont = beta1;
                    else if dqp>1
                            discont = beta2;
                        end
                    end
                    tmp = dt(i+1, k) + discont; 
                    if dt(i, j) > tmp
                        dt(i, j) = tmp;
                    end
                end
            end
            
            dt(i,j) = dt(i,j)+dist;
        end
    end
    Bv(:,col,:) = dt;
end
Cv = Fv + Bv - mv;
Cv(isnan(Cv))=inf;
Cv = Cv(2:(x-1), 2:(y-1), :);
mv = mv(2:(x-1), 2:(y-1), :);
x=x-2;
y=y-2;
%to find the V matrix, we need to first find the horizontal message flow 
%across columns, so we should use Cv matrix, define a new intermediate data
%structure called MsgV, which is two matrices, MsgVleft(x, y, 2*maxd+1),
%and MsgVright(x, y, 2*maxd+1); Also we have MsgHup, MsgHdown for later use
MsgVleft = ones(x, y, 2*maxd+1)*inf;
MsgVright = ones(x, y, 2*maxd+1)*inf;
MsgHup = ones(x, y, 2*maxd+1)*inf;
MsgHdown = ones(x, y, 2*maxd+1)*inf;
fprintf('calculating messages between lines...\n');
MsgVleft(:, y, :) = Cv(:, y, :);
for j = (y-1):-1:1
    l = repmat(MsgVleft(:, j+1, :), 1, (2*maxd+1), 1);
    d = ones(x, 2*maxd+1, 2*maxd+1)*beta2;
    d(:, 2*maxd+1, 2*maxd+1) = 0.;
    for i = 1:2*maxd
        d(:, i, i) = 0.;
        d(:, i, i+1) = beta1;
        d(:, i+1, i) = beta1;
    end
    MsgVleft(:, j, :) = reshape(min( d + l, [], 3), x, 2*maxd+1);
    MsgVleft(:, j, :) = MsgVleft(:, j, :) + Cv(:, j, :);
end
    
MsgVright(:, 1, :) = Cv(:, 1, :);
for j = 2:y
    l = repmat(MsgVright(:, j-1, :), 1, (2*maxd+1), 1);
    d = ones(x, 2*maxd+1, 2*maxd+1)*beta2;
    d(:, 2*maxd+1, 2*maxd+1) = 0.;
    for i = 1:2*maxd
        d(:, i, i) = 0.;
        d(:, i, i+1) = beta1;
        d(:, i+1, i) = beta1;
    end
    MsgVright(:, j, :) = reshape(min( d + l, [], 3), x, 2*maxd+1);
    MsgVright(:, j, :) = MsgVright(:, j, :) + Cv(:, j, :);
end

%Now we use the V matrix to modify the m(p,dp) -> m'(p,dp) and 
%run the optimization on the Horizontal Tree. find H(p,dp)
fprintf('calculating V matrices...\n');
diff = beta2 - beta1;
V = ones(x, y, 2*maxd+1) * inf;
%H = ones(x, y, 2*maxd+1) * inf;
for k = 2:(y-1)
    for j = 1:x
        for i = 1:(2*maxd+1)
            tmp = repmat(reshape(MsgVleft(j, k+1, :),2*maxd+1,1), 1, 2*maxd+1)...
                + repmat(reshape(MsgVright(j, k-1, :),1,2*maxd+1), 2*maxd+1, 1) + beta2;
            if i > 1
                tmp(i-1, :) = tmp(i-1, :) - diff;
                tmp(:, i-1) = tmp(:, i-1) - diff;
            end
            if i < (2*maxd+1)
                tmp(i+1, :) = tmp(i+1, :) - diff;
                tmp(:, i+1) = tmp(:, i+1) - diff;
            end
            tmp(i, :) = tmp(i, :) - beta2;
            tmp(:, i) = tmp(:, i) - beta2;
            
            V(j, k, i) = min(min(tmp)) + Cv(j,k,i);
        end
    end
end

fprintf('displaying images...\n');
disparity = zeros(x,y);
for i = 1:x
    for j = 1:y
        disparity(i, j) = argmin(reshape(V(i,j,:),2*maxd+1, 1));
    end
end
colors = linspace(0.,252., 2*maxd+1);
disparity = colors(disparity);
parse_img = repmat(disparity, 1,1,3)/255.;
image(parse_img);
