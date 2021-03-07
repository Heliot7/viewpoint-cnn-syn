classdef ImageNet3D
    properties
        path = 'Real\Multi\ImageNet3D\';
        % All classes: {'car', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'chair', 'bottle', 'sofa', 'bus', 'diningtable', 'train', 'tvmonitor'}
%         classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
        classes = {'aeroplane', 'bicycle', 'motorbike', 'boat', 'car', 'chair', 'bottle', 'sofa', 'bus', 'diningtable', 'train', 'tvmonitor'};        
%         classes = {'car'};
        isOcclusions = true;
        parts = {};
        % Max Viewpoints: 360
        % - Contains elevation
        azimuth = 16;
    end
end