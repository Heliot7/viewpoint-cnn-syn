function saveHOGs(folder, fileName, posHOG, negHOG)

    posHOG = sum(posHOG(:,:,:), 3);
    if(nargin > 3)
        negHOG = sum(negHOG(:,:,:), 3);
    else
        negHOG = zeros(size(posHOG,1), size(posHOG,2));
    end
    
    createDir(folder);
    
    f = figure;
    set(f, 'visible', 'off');
    subplot(1,3,1);
    imagesc(posHOG); axis off; axis image;
    title('Positive weights'); 
    subplot(1,3,2);
    imagesc(negHOG); axis off; axis image;
    title('Negative weights'); 
    subplot(1,3,3);
    imagesc(posHOG - negHOG); axis off; axis image;
    title('Diff'); 
    saveas(f, [folder fileName]);
    
end

