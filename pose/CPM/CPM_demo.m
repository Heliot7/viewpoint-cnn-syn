%% Demo code of "Convolutional Pose Machines", 
% Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
% In CVPR 2016
% Please contact Shih-En Wei at shihenw@cmu.edu for any problems or questions
%%
close all;
% addpath('src'); 
% addpath('util');
% addpath('util/ojwoodford-export_fig-5735e6d/');
param = config();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);

%% Edit this part
% put your own test image here
% test_image= 'Z:\PhD\Data\Real/Multi/PASCAL_VOC12/train/JPEGImages/2008_000185.jpg';
% test_image = 'Z:\PhD\Data\Real\Car\EPFL\tripod_seq_01_039.jpg';
% test_image = 'Z:\PhD\Data\Real/Multi/PASCAL_VOC12/train/JPEGImages/2008_000027.jpg';
% test_image = 'Z:\PhD\Data\Real/Multi/PASCAL_VOC12/train/JPEGImages/2008_000052.jpg';
% test_image = 'D:\Core\CPM\dataset\LEEDS\lsp_dataset\images\im0121.jpg';
test_image = 'Z:\PhD\Data\Real\Human\INRIA\Test\pos\person_076.png';
% test_image = 'Z:\PhD\Data\Real\Human\INRIA\Test\pos\crop_000001.png';
% test_image = 'D:/Core/CPM/testing/sample_image/singer.jpg';
% test_image = 'D:/Core/CPM/testing/sample_image/shihen.png';
% test_image = 'D:/Core/CPM/testing/sample_image/roger.png';
% test_image = 'D:/Core/CPM/testing/sample_image/roger_368.png';
%test_image = 'D:/Core/CPM/testing/sample_image/nadal.png';
%test_image = 'D:/Core/CPM/testing/sample_image/LSP_test/im1640.jpg';
%test_image = 'D:/Core/CPM/testing/sample_image/CMU_panoptic/00000998_01_01.png';
%test_image = 'D:/Core/CPM/testing/sample_image/CMU_panoptic/00004780_01_01.png';
%test_image = 'D:/Core/CPM/testing/sample_image/FLIC_test/princess-diaries-2-00152201.jpg';
interestPart = 'Rsho'; % '(3) left\_front\_wheel'; % to look across stages. check available names in config.m

%% core: apply model on the image, to get heat maps and prediction coordinates
figure(1); 
imshow(test_image);
hold on;
title('Drag a bounding box');
rectangle = getrect(1);
[heatMaps, prediction] = applyModel(test_image, param, rectangle);

%% visualize, or extract variable heatMaps & prediction for your use
visualize(test_image, heatMaps, prediction, param, rectangle, interestPart);