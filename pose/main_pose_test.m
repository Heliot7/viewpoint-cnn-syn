% varargin: test data to evaluate -> ({'gt','bb'}, {'fast','faster'})
function main_pose_test(input, nameDataset, nameTask, varargin)

    clc;
    close all;
    % clear all;
    warning ('off','all');
    
    if(nargin == 0)
        input = InputParameters;
    end
    input.typeClassifier = 'CNN';
    if(exist('nameDataset','var'))
        input.sourceDataset = eval(nameDataset);
        input.targetDataset = eval(nameDataset);
    end
    if(~exist('nameTask','var') || isempty(nameTask) || strcmpi(nameTask,''))
        if(strfind(lower(input.typeClassifier), 'cnn'))
            nameTask = input.cnnName;
        else
            nameTask = 'Pose';
        end
    end
    dataEval = 'gt';
    if(~isempty(varargin))
        dataEval = varargin{1};
        dataDet = varargin{2}; 
    end
    
    fprintf('\nMulti-Object/Pose estimation:\n');
    fprintf('Dataset: %s\n', class(input.targetDataset));
	[input, data, ~, testData] = getData(input, 'target');

    % -> Prepare output folders
    nameRun = strrep([class(input.targetDataset) '_' dataEval '_' input.cnnName '_' input.cnnModel '_' input.cnnPoseTypeCNN], '\', '_');
    mDir = [input.PATH_RESULTS nameRun];
    % removeDir(mDir);
    createDir(mDir);
    
    % For all objects 
    numClasses = length(input.sourceDataset.classes);
    for idxObj = 1:numClasses
        
        strObj = input.sourceDataset.classes{idxObj};
        nameFile = ['\' strObj]; % nameFile = ['\' nameRun '_' strObj];
        try
            load([mDir nameFile]);
        catch
            % all classes together
            if(input.jointDetPose)
                aux_obj = strObj;
                strObj = 'all';
                classId = idxObj;
                kpsId = zeros(numClasses,2);
                kpss = 0;
                for i = 1:numClasses
                    kpsId(i,1) = kpss;
                    if(isfield(data,'partLabels'))
                        kpsId(i,2) = length(data.partLabels{i})+1;
                    else
                        kpsId(i,2) = 0;
                    end
                    kpss = kpss + kpsId(i,2);
                end
            else
                classId = 1;
                if(isfield(data,'partLabels'))
                    kpsId = [0, length(data.partLabels{1})+1];
                else
                    kpsId = [0 1];
                end
            end
            % Load caffe model
            model = [input.PATH_CNN nameTask '\' lower(strObj) '\' lower(strObj) '_deploy.prototxt'];
            weights = [input.PATH_CNN nameTask '\' lower(strObj) '\' lower(strObj) '_' input.cnnModel '.caffemodel'];
            caffe.set_mode_gpu();
            caffe.set_device(1);
            netCaffe = caffe.Net(model, weights, 'test');
            sizePatch = netCaffe.blob_vec(1).shape();
            fprintf('==========\n');
            fprintf('CNN Net for %s: class %s\n', lower(nameTask), lower(strObj));
            fprintf('==========\n');
            % back to obj id
            if(input.jointDetPose)
                strObj = aux_obj;
            end
            % Detect poses from given objects (known BB)
            if(strcmpi(input.typePipeline, 'class'))
                % Take class info from metadata
                classDataGT = getDataClass(input, testData, strObj);
                % classDataGT = getDataClass(input, data, strObj);
                classData = classDataGT;
                bb_gt = classDataGT.annotations.BB;
                bb_gt = bb_gt(:,[2 1 4 3]);
                bb_gt(:,3) = bb_gt(:,3) + bb_gt(:,1);
                bb_gt(:,4) = bb_gt(:,4) + bb_gt(:,2);
                gt = [bb_gt, classDataGT.annotations.imgId];
                bb = bb_gt;
                ids = classDataGT.annotations.imgId;
                if(strcmpi(dataEval,'bb'))
                    % Take bb's from Fast R-CNN results
                    str_dataset = '';
                    if(strcmpi(class(input.targetDataset),'Pascal3D'))
                        str_dataset = 'P3D';
                    elseif(strcmpi(class(input.targetDataset),'ObjectNet3D'))
                        str_dataset = 'O3D';
                    end                
                    bb_path = [input.PATH_CNN 'FasterRCNN\output\' dataDet '_rcnn_' str_dataset '_VGG16\test\'];
                    load([bb_path strObj '_boxes_' class(input.targetDataset) '_test']);
                    inds = cell2mat(inds);
                    ids = zeros(size(inds,1),1);
                    pivotIds = 1;
                    % Select detection after NMS
                    for i = 1:length(boxes);
                        bbox = boxes{i};
                        keep = nms(bbox, 0.3);
                        bbox = bbox(keep,:);
                        numDet = size(bbox,1);
                        ids(pivotIds:pivotIds+numDet-1) = i*ones(numDet,1);
                        pivotIds = pivotIds + numDet;
                        boxes{i} = bbox;
                    end
                    bb = cell2mat(boxes);
                    scores = bb(:,5);
                    bb = bb(:,1:4);
                    ids = ids(1:pivotIds-1);
                    list_classes = repmat({strObj},[length(ids) 1]);
                    [~, si] = sort(scores, 'descend'); % sort scores
                    ids = ids(si);
                    bb = bb(si,:);
                    [AP, ~, ~, objMatches, resDetections] = evalAVP(classDataGT, gt, [bb, ids]);
                    % Also update bb annotation (x1, y1, x2, y2, score)
                    bb_pau = bb(:,[2 1 4 3]);
                    bb_pau(:,3) = bb_pau(:,3) - bb_pau(:,1);
                    bb_pau(:,4) = bb_pau(:,4) - bb_pau(:,2);
                    % Select final detections (true positives)
                    % Assign to detected BB struct
                    classData.annotations.imgId = ids(resDetections);
                    classData.annotations.BB = bb_pau(resDetections,:);
                    classData.annotations.classes = list_classes(resDetections);
                    objMatches = objMatches(objMatches > 0);
                    classData.annotations.parts = classDataGT.annotations.parts(objMatches);
                    classData.annotations.vp.azimuth = classDataGT.annotations.vp.azimuth(objMatches);
                    classData.annotations.vp.elevation = classDataGT.annotations.vp.elevation(objMatches);
                    classData.annotations.vp.distance = classDataGT.annotations.vp.distance(objMatches);
                    classData.annotations.vp.plane = classDataGT.annotations.vp.plane(objMatches);
                    classData.annotations.camera.px = classDataGT.annotations.camera.px(objMatches);
                    classData.annotations.camera.py = classDataGT.annotations.camera.py(objMatches);
                    classData.annotations.camera.focal = classDataGT.annotations.camera.focal(objMatches);
                    classData.annotations.camera.viewport = classDataGT.annotations.camera.viewport(objMatches);
                else % Remove all test viewpoints with (0,0,0) -> as Tulsiani et al.
%                     if(strcmpi(nameDataset,'Pascal3D'))
%                         % isSum3 = (classDataGT.annotations.vp.azimuth == 0) + (classDataGT.annotations.vp.elevation == 0) + (classDataGT.annotations.vp.plane == 0);
%                         % wrong = find(isSum3 == 3)';
%                         list_wrongs = {[2 6 9 37 68 76 84 91 96 97 104 110 159 172 176 180 183 187 198 199 212 226 230 242 248 268 269 270 272 280 307 308 319 326 329 333 367 393 406 408 414 418 424 430]; ... % aeroplane
%                             [16 40 63 113 117 118 139 141 169 170 178 211 237 266 279 317 325]; ... % bycicle
%                             [11 24 30 39 41 43 44 56 57 59 60 63 77 80 82 86 89 90 91 92 93 101 102 112 130 134 135 139 150 155 157 159 166 170 172 176 184 185 186 196 197 205 206 207 209 225 227 239 248 251 253 269 274 281 285 289 314 322 323 329 334 343 347 348 350 353 355 369 386 401 402 417 424]; ... % boat
%                             [6 15 88 92 111 122 123 124 125 189 190 191 279 305 319 320 321 326 328 329 343 344 360 422 432 433 445 447 470 485 493 494 500 548 553 555 558 559 569 587 588 630]; ... % bottle
%                             [28 31 37 44 47 48 56 82 84 91 102 108 113 118 134 157 158 159 166 179 184 209 248 270 271 278 283 290 293 298]; ... % bus
%                             [2 5 7 12 19 20 32 41 42 95 105 111 112 136 194 202 216 242 252 261 265 272 278 280 291 293 294 308 311 340 367 371 375 376 378 415 421 425 427 428 435 440 447 448 461 464 469 470 471 472 473 474 489 491 493 498 500 505 508 512 515 520 521 524 528 530 531 533 548 551 552 555 556 561 566 570 574 583 585 586 591 593 594 600 602 605 608 609 612 615 616 617 621 622 635 645 650 661 662 668 686 688 697 698 709 711 721 727 729 745 757 759 760 767 785 804 813 821 834 848 854 865 874 882 884 889 895 898 915 919 922 929 930 941 946 953 961 962 976 978 980 994 995 996 997 1003]; ... % car
%                             [25 38 41 45 66 77 118 136 157 199 201 208 263 264 265 279 313 316 322 374 380 384 403 404 407 445 456 457 473 476 477 486 505 508 514 531 534 542 545 556 558 567 590 612 639 640 642 655 676 683 689 740 750 799 838 845 846 848 853 857 874 890 891 917 925 933 935 940 952 959 964 976 1010 1011 1022 1027 1034 1049 1056 1061 1075 1079 1090 1093 1095 1100 1103 1116 1118 1131 1168]; ... % chair
%                             [13 17 18 23 26 32 43 45 58 62 63 64 65 66 70 73 76 81 91 93 94 104 108 110 112 115 118 127 128 132 133 135 136 142 145 147 154 156 157 159 160 162 170 172 176 177 179 180 182 186 189 191 192 197 200 202 203 204 205 206 209 213 219 221 223 224 227 228 231 238 240 243 250 252 253 256 257 258 262 265 268 273 275 279 282 289 291 294 296 298]; ... % dtable
%                             [28 50 68 70 72 90 138 158 172 188 201 202 208 216 217 218 228 247 252 268 278 304 313 316 348]; ... % mbike
%                             [2 4 19 20 22 46 66 71 82 84 87 88 92 93 95 108 109 110 115 116 118 125 126 133 145 158 159 161 163 164 167 169 172 173 175 192 194 200 203 205 217 218 221 225 233 238 239 240 253 265 266 268 271 273 276 280 282]; ... % sofa
%                             [8 16 20 31 34 45 50 58 67 73 82 105 118 130 153 163 166 171 189 211 236 261 276 298 313 315]; ... % train
%                             [30 43 54 98 124 132 146 147 150 152 168 175 199 221 228 236 238 239 240 241 242 243 293]}; % tv
%                         wrongs = list_wrongs{classId};
%                         classDataGT.annotations.imgId(wrongs) = [];
%                         classDataGT.annotations.BB(wrongs,:) = [];
%                         classDataGT.annotations.classes(wrongs) = [];
%                         classDataGT.annotations.parts(wrongs) = [];
%                         classDataGT.annotations.vp.azimuth(wrongs) = [];
%                         classDataGT.annotations.vp.elevation(wrongs) = [];
%                         classDataGT.annotations.vp.distance(wrongs) = [];
%                         classDataGT.annotations.vp.plane(wrongs) = [];
%                         classDataGT.annotations.camera.px(wrongs) = [];
%                         classDataGT.annotations.camera.py(wrongs) = [];
%                         classDataGT.annotations.camera.focal(wrongs) = [];
%                         classDataGT.annotations.camera.viewport(wrongs) = [];
%                         classData = classDataGT;
%                         bb_gt = classDataGT.annotations.BB;
%                         bb_gt = bb_gt(:,[2 1 4 3]);
%                         bb_gt(:,3) = bb_gt(:,3) + bb_gt(:,1);
%                         bb_gt(:,4) = bb_gt(:,4) + bb_gt(:,2);
%                         gt = [bb_gt, classDataGT.annotations.imgId];
%                         bb = bb_gt;
%                         ids = classDataGT.annotations.imgId;
%                     end
                end

                % Run pose estimation + plot results
                [vp, isCorrectVP, isCorrectVP_abs, isCorrect_pr, angleError, geocDist, PCK, PCK_cwm] = ...
                    poseEstimation(input, classData, netCaffe, sizePatch, classId-1, kpsId(idxObj,:)');
                % Estimate AVP
                allVP = zeros(length(ids),5);
                if(strcmpi(dataEval,'bb'))
                    allVP(resDetections,:) = isCorrectVP;
                else
                    allVP = isCorrectVP;
                end
                [~, AVP] = evalAVP(classDataGT, gt, [bb, ids], allVP);
                VP = sum(isCorrectVP) / length(isCorrectVP);
                if(strcmpi(dataEval,'bb'))
                    allVP(resDetections,:) = isCorrectVP_abs;
                else
                    allVP = isCorrectVP_abs;
                end
                [AP, AVP_abs] = evalAVP(classDataGT, gt, [bb, ids], allVP);
                VP_abs = sum(isCorrectVP_abs) / length(isCorrectVP_abs);
                % Estimate VP_pr
                vp_pr = zeros(4,1);
                for idxVP = 1:4
                    vp_aux = isCorrect_pr{idxVP};
                    all_pr = vp_aux(:,2) ./ sum(vp_aux,2);
                    all_pr(isnan(all_pr)) = [];
                    vp_pr(idxVP) = mean(all_pr);
                end
                
                % Estimate median angle error
                medError = median(angleError);
                % Estimate mean angle error
                meanError = mean(angleError);
                % Estimate median error based on geodesic distance
                medError3D = median(geocDist);
                % Estimate PCK
                PCK = PCK(:,1) ./ PCK(:,2);
                PCK_cwm(:,1:5) = PCK_cwm(:,1:5) ./ PCK_cwm(1,6);
                
            elseif(strcmpi(input.typePipeline, 'detection'))
                % Run Fast R-CNN for object detection
            end
            
            % Save results in mat/txt files
            detections = [];
            % -> detection results
            detections.bbox = bb;
            detections.ids = ids;
            detections.vp = vp;
            % -> AP and AVPs
            detections.AP = AP*100;
            detections.AVP = AVP*100;
            detections.VP = VP*100;
            detections.AVP_abs = AVP_abs*100;
            detections.VP_abs = VP_abs*100;
            detections.VP_pr = vp_pr*100;
            detections.medError = medError;
            detections.meanError = meanError;
            detections.medError3D = medError3D;
            detections.PCK = PCK;
            detections.PCK_cwm = PCK_cwm;
            save([mDir nameFile], 'detections', '-v7.3');
            
            % Remove caffe instance
            caffe.reset_all();
        end

        meanPCK = mean(detections.PCK * 100);
        meanPCK_cwm = mean(detections.PCK_cwm * 100);
        nameTxtFile = [mDir '\' strObj ...
            '_acc3D_' sprintf('%.1f',detections.VP_abs(5)) ...
            '_err3D_' sprintf('%.1f',detections.medError3D) ...
            '_PCK_' sprintf('%.1f', meanPCK) ...
            '_AVP24_' sprintf('%.1f', detections.AVP(4)) '.txt'];
        % nameTxtFile = [mDir nameFile '_AP_' sprintf('%.1f',detections.AP) '_AVP24_' sprintf('%.1f', detections.AVP(4)) '.txt'];
        fid = fopen(nameTxtFile,'w');
        fclose(fid);
        diary(nameTxtFile);
        diary on;
        fprintf('%s\n', nameFile);
        fprintf('[%s] AP = %.1f\n', strObj, detections.AP);
        fprintf('[%s] AVP04 = %.1f (VP: %.1f, VP_pr: %.1f) AVP04_abs = %.1f (VP_abs: %.1f)\n', strObj, detections.AVP(1), detections.VP(1), detections.VP_pr(1), detections.AVP_abs(1), detections.VP_abs(1));
        fprintf('[%s] AVP08 = %.1f (VP: %.1f, VP_pr: %.1f) AVP08_abs = %.1f (VP_abs: %.1f)\n', strObj, detections.AVP(2), detections.VP(2), detections.VP_pr(2), detections.AVP_abs(2), detections.VP_abs(2));
        fprintf('[%s] AVP16 = %.1f (VP: %.1f, VP_pr: %.1f) AVP16_abs = %.1f (VP_abs: %.1f)\n', strObj, detections.AVP(3), detections.VP(3), detections.VP_pr(3), detections.AVP_abs(3), detections.VP_abs(3));
        fprintf('[%s] AVP24 = %.1f (VP: %.1f, VP_pr: %.1f) AVP24_abs = %.1f (VP_abs: %.1f)\n', strObj, detections.AVP(4), detections.VP(4), detections.VP_pr(4), detections.AVP_abs(4), detections.VP_abs(4));
        fprintf('[%s] AVP3D = %.1f (VP: %.1f) AVP3D_abs = %.1f (VP_abs: %.1f)\n', strObj, detections.AVP(5), detections.VP(5), detections.AVP_abs(5), detections.VP_abs(5));
        fprintf('[%s] medError = [a %.1f, e %.1f, p %.1f] \n', strObj, detections.medError(1), detections.medError(2), detections.medError(3));
        fprintf('[%s] meanError = [a %.1f, e %.1f, p %.1f] \n', strObj, detections.meanError(1), detections.meanError(2), detections.meanError(3));
        fprintf('[%s] medError3D = %.1f \n', strObj, detections.medError3D);
        fprintf('[%s] PCK', strObj);
        for i = 1:length(detections.PCK)
            fprintf(' %.1f', detections.PCK(i) * 100.0);
        end
        fprintf('\n');
        fprintf('[%s] avg PCK %.1f\n', strObj, meanPCK);
        
        fprintf('[%s] PCK_cwm:\n', strObj);
        notation_cwm = {'(OK) found', '(OK) found_occ', '(NO) misplaced', '(NO) missed', '(NO) should not be there'};
        for n = 1:5
            fprintf('%s: %.1f\n', notation_cwm{n}, meanPCK_cwm(n));
            for i = 1:length(detections.PCK_cwm)
                fprintf('%.1f ',  detections.PCK_cwm(i,n) * 100.0);
            end
            fprintf('\n');
        end
        diary off;
    end
    
end

