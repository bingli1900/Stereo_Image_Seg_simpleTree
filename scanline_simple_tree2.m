maxd = 18;
lambda = 0.06; %0.03
view1 = view1(:, 2:(y-1), :);
view2 = view2(:, 2:(y-1), :);
V = V(:, 2:(y-1), :);
y = y-2;

%V(x,y,2*maxd+1)is the vertical connection matrix from 1st part
minV = repmat(min(V, [], 3), 1, 1, 2*maxd+1);
V = lambda*(V - minV);
beta1 = 1.;
beta2 = 5.;
ratio = 0.001;
beta1 = beta1*ratio;
beta2 = beta2*ratio;

Fh = ones(x, y, 2*maxd+1) * inf;
Bh = ones(x, y, 2*maxd+1) * inf;
mh = ones(x, y, 2*maxd+1) * inf;

%Now calculate the Horizontal matrix Fh and Bh
fprintf('calculating horizontal scanlines...\n');
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
    end
    mh(row, y, :) = dt(y, :);
    mh(row, y, :) = mh(row, y, :) + V(row, y, :);
    dt(y, :) = mh(row, y, :);
    
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
            dist = dist + V(row, i, j);
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

for row = 1:x
    dt = ones(y, 2*maxd+1) * inf;
    line1 = reshape(view1(row,:,:),y,3);
    line2 = reshape(view2(row,:,:),y,3);
    %initialize the first line of the dynamic table, the leftmost pixel
    dt(1, :) = mh(row, 1, :);
    
    %now start to update the dynamic table
    for i = 2:y
        for j = 1:(2*maxd+1)
            dq = j - maxd - 1;
            if i+dq<1 || i+dq>y
                continue;
            end
            dist = mh(row, i, j);
            
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
Ch = Bh + Fh - mh;
Ch(isnan(Ch))=inf;

fprintf('calculating messages between lines...\n');
MsgHup = ones(x, y, 2*maxd+1)*inf;
MsgHdown = ones(x, y, 2*maxd+1)*inf;
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

diff = beta2 - beta1;
H = ones(x, y, 2*maxd+1) * inf;
fprintf('calculating for H matrix.\n');
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

fprintf('displaying disparity image using H matrix...\n');
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
