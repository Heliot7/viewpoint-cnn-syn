function saveCNNconv5(folder, fileName, w_pos, w_neg)

    % Group 256 filters of conv 5 in a single image
    posImg = zeros(1024);
    negImg = zeros(1024);
    numDim = int32(sqrt(256));
    size = int32(1024 / numDim);
    for r = 1:numDim
        for c = 1:numDim
            filter = imresize(reshape(w_pos(:,:,(r-1)*numDim+c),[13 13]), [size size]);
            filter(1,:) = 0; filter(end,:) = 0; filter(:,1) = 0; filter(:,end) = 0;
            posImg((r-1)*size+1:(r-1)*size+size, (c-1)*size+1:(c-1)*size+size) = filter;
            filter = imresize(reshape(w_neg(:,:,(r-1)*numDim+c),[13 13]), [size size]);
            negImg((r-1)*size+1:(r-1)*size+size, (c-1)*size+1:(c-1)*size+size) = filter;
        end
    end
    
    createDir(folder);
    
    f = figure;
    set(f, 'visible', 'off');
    subplot(1,2,1);
    imagesc(posImg); axis off; axis image;
    title('Positive weights'); 
    subplot(1,2,2);
    imagesc(negImg); axis off; axis image;
    title('Negative weights'); 
    saveas(f, [folder fileName '.png']);

end

