function param = config()
%% set this part

% CPU mode (0) or GPU mode (1)
% friendly warning: CPU mode may take a while
param.use_gpu = 1;

% GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 0;

% Select model (default: 1)
% 1: MPII+LSP(PC) 6-stage CPM
% 2: MPII 6-stage CPM
% 3: LSP(PC) 6-stage CPM
% 4: FLIC 4-stage CPM (upper body only)
param.modelID = 3;

% Scaling paramter: starting and ending ratio of person height to image
% height, and number of scales per octave
% warning: setting too small starting value on non-click mode will take
% large memory
param.octave = 6;


%% don't edit this part

% path of your caffe
%caffepath = textread('../caffePath.cfg', '%s', 'whitespace', '\n\t\b ');
%caffepath= [caffepath{1} '/matlab/'];
%fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
%addpath(caffepath);
caffe.reset_all();
if(param.use_gpu)
    fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
    caffe.set_mode_gpu();
    caffe.set_device(GPUdeviceNumber);
else
    fprintf('Setting to CPU mode.\n');
    caffe.set_mode_cpu();
end

param.click = 1;

param.model(1).caffemodel = 'D:/Core/CPM/model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel';
param.model(1).deployFile = 'D:/Core/CPM/model/_trained_MPI/pose_deploy_centerMap.prototxt';
param.model(1).description = 'MPII+LSP 6-stage CPM';
param.model(1).description_short = 'MPII_LSP_6s';
param.model(1).boxsize = 368;
param.model(1).padValue = 128;
param.model(1).np = 14;
param.model(1).sigma = 21;
param.model(1).stage = 6;
param.model(1).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(1).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(2).caffemodel = 'D:/Core/CPM/model/_trained_MPI/pose_iter_630000.caffemodel';
param.model(2).deployFile = 'D:/Core/CPM/model/_trained_MPI/pose_deploy_centerMap.prototxt';
param.model(2).description = 'MPII 6-stage CPM';
param.model(2).description_short = 'MPII_6s';
param.model(2).boxsize = 368;
param.model(2).padValue = 128;
param.model(2).np = 14;
param.model(2).sigma = 21;
param.model(2).stage = 6;
param.model(2).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(2).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
% param.model(3).caffemodel = 'D:/Core/CPM/model/_trained_LEEDS_PC/pose_iter_395000.caffemodel';
param.model(3).caffemodel = 'D:/Core/CPM/model/_trained_LEEDS_PC/pose_iter_75000.caffemodel';
param.model(3).deployFile = 'D:/Core/CPM/model/_trained_LEEDS_PC/pose_deploy_centerMap.prototxt';
param.model(3).description = 'LSP (PC) 6-stage CPM';
param.model(3).description_short = 'LSP_6s';
param.model(3).boxsize = 368;
param.model(3).np = 14;
param.model(3).sigma = 21;
param.model(3).stage = 6;
param.model(3).padValue = 128;
param.model(3).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(3).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(4).caffemodel = 'D:/Core/CPM/model/_trained_FLIC/pose_iter_40000.caffemodel';
param.model(4).deployFile = 'D:/Core/CPM/model/_trained_FLIC/pose_deploy.prototxt';
param.model(4).description = 'FLIC (upper body only) 4-stage CPM';
param.model(4).description_short = 'FLIC_4s';
param.model(4).boxsize = 368;
param.model(4).np = 9;
param.model(4).sigma = 21;
param.model(4).stage = 4;
param.model(4).padValue = 128;
param.model(4).limbs = [1 2; 2 3; 4 5; 5 6];
param.model(4).part_str = {'Lsho', 'Lelb', 'Lwri', ...
                           'Rsho', 'Relb', 'Rwri', ...
                           'Lhip', 'Rhip', 'head', 'bkg'};
                       
param.model(5) = param.model(4);
param.model(5).caffemodel = 'D:/Core/CPM/model/_trained_FLIC/pose_iter_50000.caffemodel';
param.model(5).deployFile = 'D:/Core/CPM/training/prototxt/FLIC/pose_deploy.prototxt';
param.model(5).part_str = {'bkg', 'head', 'Rhip', ...
                           'Lhip', 'Rwri', 'Relb', ...
                           'Rsho', 'Lwri', 'Lelb', 'Lsho'};
param.model(6) = param.model(5);
param.model(6).caffemodel = 'D:/Core/CPM/model/_trained_FLIC/pose_iter_250000.caffemodel';
param.model(7) = param.model(5);
param.model(7).caffemodel = 'D:/Core/CPM/model/_trained_FLIC/pose_iter_500000.caffemodel';

param.model(8).caffemodel = 'Z:/PhD/Data/CNN/Pose/caffemodel/pose_car_iter_100000.caffemodel';
% param.model(8).caffemodel = 'Z:/PhD/Data/CNN/Pose/caffemodel/pose_car_iter_6000_FT.caffemodel';
param.model(8).deployFile = 'Z:/PhD/Data/CNN/Pose/pose_car_deploy.prototxt';
% param.model(8).deployFile = 'Z:/PhD/Data/CNN/Pose/pose_car_deploy_FT.prototxt';
param.model(8).description = 'Test CPM Car - Pascal3D';
param.model(8).description_short = 'Test CPM Car';
param.model(8).boxsize = 368;
param.model(8).padValue = 128;
param.model(8).np = 12;
param.model(8).sigma = 21;
param.model(8).stage = 6;
sets = {(1:12)', (1:12)'};
[xCon, yCon] = ndgrid(sets{:});
allConn = [xCon(:) yCon(:)];
allConn(allConn(:,1) == allConn(:,2),:) = [];
param.model(8).limbs = allConn;
param.model(8).part_str = {'(1) right\_front\_wheel', '(2) right\_back\_wheel', '(3) left\_front\_wheel', '(4) left\_back\_wheel', ...
    '(5) upper\_right\_windshield', '(6) upper\_left\_windshield', '(7) upper\_left\_rearwindow', '(8) upper\_right\_rearwindow', ...
    '(9) right\_front\_light', '(10) left\_front\_light', '(11) left\_back\_trunk', '(12) right\_back\_trunk', '(B) background'};

