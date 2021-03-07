function temp_plots2()

    %% Num 3D models
    figure;
%     listModels = [1 2 3 5 7 10 15];
    listNeg = [10, 50, 100, 250, 500];
%     AP_neg_tgt = [14.8 45.2 64.8 67.8 70.6];
    AP_neg_tgt = [3.5 25.7 31.7 46.4 63.8];
%     AP_neg_both = [40.1 50.8 61.2 65.5 69.2];
	AP_neg_both = [34.7 48.8 51.7 59.4 59.5];
%     AP_syn = [30.3 30.3 30.3 30.3 30.3];
    AP_syn = [43.9 43.9 43.9 43.9 43.9];
    
%     EPFL_8_no = [80.83 84.77 87.9 88.53 89.63 88.98 89.29];
%     EPFL_8_da = [90.68 91.81 93.34 91.67 91.74 91.88 91.73];
%     EPFL_16_no = [64.64 70.44 73.00 76.42 76.16 76.1 77.69];
%     EPFL_16_da = [79.14 80.25 80.28 82.81 83.53 83.14 81.86];
%     EPFL_24_no = [57.81 60.80 62.02 65.68 67.74 66.11 67.59];
%     EPFL_24_da = [67.34 67.65 70.64 73.14 73.86 71.85 72.93];
% 
%     PASCAL_8_no = [75.02 77.53 80.19 79.68 80.44 79.72 79.02];
%     PASCAL_8_da = [79.74 81.49 82.32 82.91 83.03 82.94 83.29];
%     PASCAL_16_no = [60.31 66.75 69.73 67.4 67.88 64.88 66.16];
%     PASCAL_16_da = [63.68 65.45 69.09 69.51 67.61 68.52 67.01];
%     PASCAL_24_no = [38.38 42.26 44.77 51.13 48.58 48.56 51.08];
%     PASCAL_24_da = [44.66 45.71 48.60 54.17 52.98 53.08 52.64];
%     
%     IMAGENET_8_no = [83.23 87.74 89.48 87.94 88.19 89.2 88.68];
%     IMAGENET_8_da = [88.66 89.44 90.23 89.95 89.85 89.99 90.45];
%     IMAGENET_16_no = [69.55 70.21 71.59 72.49 72.53 72.39 74.10];
%     IMAGENET_16_da = [69.55 71.56 72.83 72.99 72.57 73.12 74.34];
%     IMAGENET_24_no = [49.47 53.64 56.03 58.75 59.09 60.57 60.73];
%     IMAGENET_24_da = [56.84 58.09 59.24 60.63 60.14 61.5 63.52];
    
    % clusters = [8, 16, 32, 64, 132, 164, 200, 240, 272, 312, 344, 400, 440, 480];
    % acc_clusters = [64.80, 63.00, 67.00, 68.50, 70.50, 70.80, 70.95, 71.50, 71.89, 72.00, 71.75, 72.15, 71.90 ,71.85];
    % numTgt = [1, 2, 5, 10, 15, 20, 25, 35, 50, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 197, 225, 250];
    % accNumTgt = [57.0, 56.0, 53.0, 56.0, 60.0, 62.0, 63.0, 67.0, 67.5, 68.0, 69.2, 69.0, 69.2, 69.0, 70.3, 70.7, 71.1, 71.5, 71.3, 72.1, 71.7, 71.5];
    
    hold on;
    grid on;
    plot(listNeg, AP_neg_tgt, '-', 'Color',[0/255, 0/255 200/255], 'LineWidth', 3);
    plot(listNeg, AP_neg_both, '-', 'Color',[0/255, 200/255 0/255], 'LineWidth', 3);
    plot(listNeg, AP_syn, '-', 'Color',[200/255, 0/255 0/255], 'LineWidth', 3);
    
%     plot(listModels, IMAGENET_8_no, '--', 'Color',[191/255, 26/255 26/255], 'LineWidth', 3, 'LineSmoothing', 'off');
%     plot(listModels, IMAGENET_8_da, '-', 'Color',[191/255, 26/255 26/255], 'LineWidth', 3, 'LineSmoothing', 'off');
%     plot(listModels, IMAGENET_16_no, '--', 'Color',[26/255, 26/255 191/255], 'LineWidth', 3, 'LineSmoothing', 'off');
%     plot(listModels, IMAGENET_16_da, '-', 'Color',[26/255, 26/255 191/255], 'LineWidth', 3, 'LineSmoothing', 'off');
%     plot(listModels, IMAGENET_24_no, '--', 'Color',[26/255, 191/255 26/255], 'LineWidth', 3, 'LineSmoothing', 'off');
%     plot(listModels, IMAGENET_24_da, '-', 'Color',[26/255, 191/255 26/255], 'LineWidth', 3, 'LineSmoothing', 'off');
    % set(gca, 'XTick', [0 20 40 60 80 100 120 140 160 180]);
    axis([0 500 0 100]);
    % axis off;
    % title('16 viewpoint refinement accuracy (EPFL dataset)');
    % xlabel('number of synthetised 3D models');
    % ylabel('viewpoints accuracy');
    
    hold off;
    
    %% Others...
    
end