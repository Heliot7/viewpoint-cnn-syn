function parserO3D()

    % Load test data ObjectNet3D
    classes = getO3D();
    input = InputParameters;
    input.sourceDataset = ObjectNet3D;
    input.sourceDataset.classes = classes;
    input.targetDataset = ObjectNet3D;
    input.targetDataset.classes = classes;
    % [input, ~, ~, testData] = getData(input, 'target');
    
    path_det = [input.PATH_DATA 'Real\Multi\ObjectNet3D\vgg16_fast_rcnn_view_objectnet3d_selective_search_iter_160000'];
    path_names = [input.PATH_DATA 'Real\Multi\ObjectNet3D\Image_sets\val.txt'];
    fid = fopen(path_names,'r');
    fileData = textscan(fid, '%s');
    names = sort_nat(fileData{1});
    fclose(fid);
    
    % Order:
    % -> name_file bb(1:4) score azimuth elevation plane (in rad -180to180)
    for i = 1:length(classes)
        
        try
            load([path_det '\detections_' classes{i}])
        catch
            % -> get GT data
            classDataGT = getDataClass(input, testData, classes{i});
            bb_gt = classDataGT.annotations.BB;
            bb_gt = bb_gt(:,[2 1 4 3]);
            bb_gt(:,3) = bb_gt(:,3) + bb_gt(:,1);
            bb_gt(:,4) = bb_gt(:,4) + bb_gt(:,2);
            vp_gt = classDataGT.annotations.vp;
            vp_gt = [vp_gt.azimuth vp_gt.elevation vp_gt.plane];
            gt = [bb_gt, classDataGT.annotations.imgId];
            % -> Get detections data
            fid = fopen([path_det '\detections_' classes{i} '.txt'], 'r');
            fileData = textscan(fid, '%s %f %f %f %f %f %f %f %f');
            fclose(fid);
            paths = fileData{1};
            [paths, posPaths] = sort_nat(paths);
            bbox = cell2mat(fileData(2:6));
            bbox = bbox(posPaths,:);
            vp = cell2mat(fileData(7:9));
            vp = vp(posPaths,:);
            bbox_nms = [];
            vp_nms = [];
            ids = [];
            pvt_paths = 1;
            len_paths = length(paths);
            len_names = length(names);
            for n = 1:len_names
                if(pvt_paths > len_paths)
                    break;
                end
                first = true; found = true;
                last_pvt = pvt_paths;
                while ((found || first) && pvt_paths <= len_paths)
                    first = false;
                    found = strcmpi(paths{pvt_paths}, names{n});
                    if (found)
                        pvt_paths = pvt_paths + 1;
                    end
                end
                currentImg = last_pvt:pvt_paths-1;
                if(~isempty(currentImg))
                    img_bbox = bbox(currentImg,:);
                    keep = nms(img_bbox, 0.3);
                    bbox_nms = [bbox_nms; img_bbox(keep,:)];
                    img_vp = vp(currentImg,:);
                    vp_nms = [vp_nms; img_vp(keep,:)];
                    ids = [ids; n*ones(length(keep),1)];
                end
            end
            scores = bbox_nms(:,5);
            [~, si] = sort(scores, 'descend'); % sort scores
            ids = ids(si);
            bbox_nms = bbox_nms(si,1:4);
            vp_nms = vp_nms(si,:); % take only azimuth
            [AP, ~, ~, objMatches, resDetections] = evalAVP(classDataGT, gt, [bbox_nms, ids]);
            vp_nms = vp_nms(resDetections,:);
            objMatches = objMatches(objMatches > 0);
            % Check VP correctness
            isCorrectVP = zeros(length(objMatches),5);
            isCorrectVP_abs = zeros(length(objMatches),5);
            angleError = zeros(length(objMatches),3);
            for idx = 1:length(objMatches)
                vp_gt_idx = vp_gt(objMatches(idx),:);
                vp_res_idx = vp_nms(idx,:)*180/pi;
                if(vp_res_idx(1) < 0)
                    vp_res_idx(1) = vp_res_idx(1) + 360;
                end
                % Compute VP4-24
                [isCorrectVP(idx,:), isCorrectVP_abs(idx,:), angleError(idx,:)] = getAVP(vp_gt_idx, vp_res_idx, 'degree');
            end
            allVP = zeros(length(ids),5);
            allVP(resDetections,:) = isCorrectVP;
            [~, AVP] = evalAVP(classDataGT, gt, [bbox_nms, ids], allVP);
            VP = sum(isCorrectVP) / length(isCorrectVP);
            allVP = zeros(length(ids),5);
            allVP(resDetections,:) = isCorrectVP_abs;
            [AP, AVP_abs] = evalAVP(classDataGT, gt, [bbox_nms, ids], allVP);
            VP_abs = sum(isCorrectVP_abs) / length(isCorrectVP_abs);
            % Estimate median angle error
            medError = median(angleError);

            % Store all information in mat file
            detections = [];
            % -> detection results
            detections.bbox = bbox_nms;
            detections.ids = ids;
            detections.vp = vp_nms;
            % -> AP and AVPs
            detections.AP = AP*100;
            detections.AVP = AVP*100;
            detections.VP = VP*100;
            detections.AVP_abs = AVP_abs*100;
            detections.VP_abs = VP_abs*100;
            detections.medError = medError;
            save([path_det '\detections_' classes{i}], 'detections', '-v7.3');
            
        end
        
        % Pose estimation results
        nameTxtFile = [path_det '\detections_' classes{i} '_AP_' sprintf('%.1f',detections.AP) '_AVP24_' sprintf('%.1f', detections.AVP(4)) '.txt'];
        fid = fopen(nameTxtFile,'w');
        fclose(fid);
        diary(nameTxtFile);
        diary on;
        fprintf('[%s] AP = %.1f\n', classes{i}, detections.AP);
        fprintf('[%s] AVP04 = %.1f (VP: %.1f) AVP04_abs = %.1f (VP_abs: %.1f)\n', classes{i}, detections.AVP(1), detections.VP(1), detections.AVP_abs(1), detections.VP_abs(1));
        fprintf('[%s] AVP08 = %.1f (VP: %.1f) AVP08_abs = %.1f (VP_abs: %.1f)\n', classes{i}, detections.AVP(2), detections.VP(2), detections.AVP_abs(2), detections.VP_abs(2));
        fprintf('[%s] AVP16 = %.1f (VP: %.1f) AVP16_abs = %.1f (VP_abs: %.1f)\n', classes{i}, detections.AVP(3), detections.VP(3), detections.AVP_abs(3), detections.VP_abs(3));
        fprintf('[%s] AVP24 = %.1f (VP: %.1f) AVP24_abs = %.1f (VP_abs: %.1f)\n', classes{i}, detections.AVP(4), detections.VP(4), detections.AVP_abs(4), detections.VP_abs(4));
        fprintf('[%s] AVP3D = %.1f (VP: %.1f) AVP3D_abs = %.1f (VP_abs: %.1f)\n', classes{i}, detections.AVP(5), detections.VP(5), detections.AVP_abs(5), detections.VP_abs(5));
        fprintf('[%s] medError = [a %.1f, e %.1f, p %.1f] \n', classes{i}, detections.medError(1), detections.medError(2), detections.medError(3));
        diary off;

    end
    
end

function classes = getO3D()

    classes = {'aeroplane', 'ashtray', 'backpack', 'basket', ...
    'bed', 'bench', 'bicycle', 'blackboard', 'boat', 'bookshelf', ...
    'bottle', 'bucket', 'bus', 'cabinet', 'calculator', 'camera', ...
    'can', 'cap', 'car', 'cellphone', 'chair', 'clock', ...
    'coffee_maker', 'comb', 'computer', 'cup', 'desk_lamp', ...
    'diningtable', 'dishwasher', 'door', 'eraser', 'eyeglasses', ...
    'fan', 'faucet', 'filing_cabinet', 'fire_extinguisher', ...
    'fish_tank', 'flashlight', 'fork', 'guitar', 'hair_dryer', ...
    'hammer', 'headphone', 'helmet', 'iron', 'jar', 'kettle', 'key', ...
    'keyboard', 'knife', 'laptop', 'lighter', 'mailbox', ...
    'microphone', 'microwave', 'motorbike', 'mouse', 'paintbrush', ...
    'pan', 'pen', 'pencil', 'piano', 'pillow', 'plate', 'pot', ...
    'printer', 'racket', 'refrigerator', 'remote_control', 'rifle', ...
    'road_pole', 'satellite_dish', 'scissors', 'screwdriver', 'shoe', ...
    'shovel', 'sign', 'skate', 'skateboard', 'slipper', 'sofa', ...
    'speaker', 'spoon', 'stapler', 'stove', 'suitcase', 'teapot', ...
    'telephone', 'toaster', 'toilet', 'toothbrush', 'train', ...
    'trash_bin', 'trophy', 'tub', 'tvmonitor', 'vending_machine', ...
    'washing_machine', 'watch', 'wheelchair'};
%     classes = {'car'};
    % -> 50 classes ShapeNet
%     classes = { 'aeroplane', 'backpack', 'basket', ...
%     'bed', 'bench', 'bicycle', 'boat', 'bookshelf', ...
%     'bottle', 'bus', 'cabinet', 'camera', ...
%     'can', 'cap', 'car', 'cellphone', 'chair', 'clock', ...
%     'desk_lamp', 'diningtable', 'dishwasher', ...
%     'faucet', 'filing_cabinet', 'guitar', ...
%     'headphone', 'helmet', 'jar', ...
%     'keyboard', 'knife', 'laptop', 'mailbox', ...
%     'microphone', 'microwave', 'motorbike', ...
%     'piano', 'pillow', 'pot', ...
%     'printer', 'remote_control', 'rifle', ...
%     'skateboard', 'sofa', 'speaker', 'stove', ...
%     'telephone', 'train', 'trash_bin', 'tub', ...
%     'tvmonitor', 'washing_machine'};
    % -> 12 classes Pascal3D+
%     classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', ...
%         'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};

end