%%
%we can use the view1 as the reference frame, then find the disparity map
%for each pixel, which minimizes the energy function. The energy function
%is made of 2 parts, similarity + smoothness

%%
%initialization of parameters
maxd = 19;
beta1 = 1.;
beta2 = 6.5;
ratio = 0.0003;
rescale = 5;
global_shift = 150;

beta1 = beta1*ratio;
beta2 = beta2*ratio;
gamma = gamma*ratio;

%read in the original images, view1 is seen from 
%the left and view2 is seen from the right 
view1 = im2double(imread('view1.png'));
view2 = im2double(imread('view5.png'));
ground_truth1 = imread('disp1.png');
ground_truth2 = imread('disp5.png');
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

Fh = ones(x, y, 2*maxd+1) * inf;
Bh = ones(x, y, 2*maxd+1) * inf;
mh = ones(x, y, 2*maxd+1) * inf;

fprintf('calculating scanlines...\n');
%Now calculate the Horizontal matrix Fh and Bh
for row = 1:x
    dt = ones(y, 2*maxd+1) * inf;
    line1 = reshape(view1(row,:,:),y,3);
    line2 = reshape(view2(row,:,:),y,3);
    %initialize the first line of the dynamic table, the leftmost pixel
    for j = (maxd+1):(2*maxd+1)
        dp = j-maxd-1;
        c1 = line1(1, :);
        c2 = line2(1+dp,:);
        dt(1, j) = colordistance(c1, c2);
    end
    
    %now start to update the dynamic table
    for i = 2:y
        c1 = line1(i, :);
        for j = 1:(2*maxd+1)
            dq = j - maxd - 1;
            if i+dq<1 || i+dq>y
                continue;
            end
            c2 = line2(i+dq,:);
            dist = colordistance(c1, c2);
            
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
    Fh(row,:,:) = dt;
end

for row = 1:x
    dt = ones(y, 2*maxd+1) * inf;
    line1 = reshape(view1(row,:,:),y,3);
    line2 = reshape(view2(row,:,:),y,3);
    %initialize the first line of the dynamic table, the leftmost pixel
    for j = 1:(maxd+1)
        dp = j-maxd-1;
        c1 = line1(y, :);
        c2 = line2(y+dp,:);
        dt(y, j) = colordistance(c1, c2);
        mh(row, y, j) = dt(y, j);
    end
    
    %now start to update the dynamic table
    for i = (y-1):-1:1
        c1 = line1(i, :);
        for j = 1:(2*maxd+1)
            dq = j - maxd-1;
            if i+dq<1 || i+dq>y
                continue;
            end
            c2 = line2(i+dq,:);
            dist = colordistance(c1, c2);
            mh(row, i, j) = dist;
            
            for k = 1:(2*maxd+1)
                dp = k - maxd-1;
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
    Bh(row,:,:) = dt;
end
Ch = Bh + Fh - mh;
Ch(isnan(Ch))=inf;

%to find the V matrix, we need to first find the horizontal message flow 
%across columns, so we should use Cv matrix, define a new intermediate data
%structure called MsgV, which is two matrices, MsgVleft(x, y, 2*maxd+1),
%and MsgVright(x, y, 2*maxd+1); Also we have MsgHup, MsgHdown for later use
MsgHup = ones(x, y, 2*maxd+1)*inf;
MsgHdown = ones(x, y, 2*maxd+1)*inf;
fprintf('calculating messages between lines...\n');

MsgHup(x, :, :) = Ch(x, :, :);
for j = (x-1):-1:1
    l = repmat(MsgHup(j+1, :, :), (2*maxd+1),1,1);
    d = ones(2*maxd+1, y, 2*maxd+1)*beta2;
    d(2*maxd+1, :, 2*maxd+1) = 0.;
    
    for i = 1:2*maxd
        d(i, :, i) = 0.;
        d(i, :, i+1) = beta1;
        d(i+1,:, i) = beta1;
    end
    MsgHup(j,:,:) = reshape(permute(min( d + l, [], 3), [2,1,3]), y, 2*maxd+1);
    MsgHup(j,:,:) = MsgHup(j, :, :) + reshape(Ch(j, :, :), 1, y, 2*maxd+1);
end

MsgHdown(1, :, :) = Ch(1, :, :);
for j = 2:x
    l = repmat(MsgHdown(j-1, :, :), (2*maxd+1),1,1);
    d = ones(2*maxd+1, y, 2*maxd+1)*beta2;
    d(2*maxd+1, :, 2*maxd+1) = 0.;
    for i = 1:2*maxd
        d(i, :, i) = 0.;
        d(i, :, i+1) = beta1;
        d(i+1,:, i) = beta1;
    end
    MsgHdown(j,:,:) = reshape(permute(min( d + l, [], 3), [2,1,3]), y, 2*maxd+1);
    MsgHdown(j,:,:) = MsgHdown(j, :, :) + reshape(Ch(j, :, :), 1, y, 2*maxd+1);
end

%Now we use the V matrix to modify the m(p,dp) -> m'(p,dp) and 
%run the optimization on the Horizontal Tree. find H(p,dp)
fprintf('calculating H matrices...\n');
diff = beta2 - beta1;
V = ones(x, y, 2*maxd+1) * inf;
H = ones(x, y, 2*maxd+1) * inf;
for k = 2:(x-1)
    for j = 1:y
        for i = 1:(2*maxd+1)
            tmp = repmat(reshape(MsgHup(k+1, j, :),2*maxd+1,1), 1, 2*maxd+1)...
                + repmat(reshape(MsgHdown(k-1, j, :),1,2*maxd+1), 2*maxd+1, 1) + beta2;
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
            
            H(k, j, i) = min(min(tmp)) + Ch(k,j,i);
        end
    end
end

fprintf('displaying images...\n');
disparity = zeros(x,y);
for i = 1:x
    for j = 1:y
        disparity(i, j) = argmin(reshape(H(i,j,:),2*maxd+1, 1));
    end
end
colors = linspace(0.,252., 2*maxd+1);
disparity = colors(disparity);
parse_img = repmat(disparity, 1,1,3)/255.;
image(parse_img);
