classdef Pascal3D
    properties
        path = 'Real\Multi\PASCAL3D\';
        % All classes: {'car', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'chair', 'bottle', 'sofa', 'bus', 'diningtable', 'train', 'tvmonitor'}
%         classes = {'bicycle', 'motorbike', 'boat', 'car', 'chair', 'bottle', 'sofa', 'bus', 'diningtable', 'train', 'tvmonitor'};
        classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
%         classes = {'car'};
        isOcclusions = true;
        parts = {};
        addImageNet3D = false;
        addKps = true;
        % Max Viewpoints: 360
        % - Contains elevation
        azimuth = 16;
    end
end