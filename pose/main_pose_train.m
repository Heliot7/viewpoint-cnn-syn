function main_pose_train(input)

    clc;
    close all;
    warning ('off','all');

    if(nargin == 0)
        input = InputParameters;
    end
    
    % PC info
    gpuID = 0;
    
    % Caffe & weights info
    input.PATH_CAFFE = 'Z:/Core/Caffe-git/';
    pathWeights = 'Z:/Core/CPM/model/_trained_MPI/pose_iter_320000.caffemodel';
    
    % Specific parameters for viewpoint estimation
    % - 224x224 batch:
    % CPM - 32/10
    % CPM - 16+16/10
    % CPM+VP - 20/6
    % CPM+VP - 10+10/6
    % CPM+VP - 5+5+10/6
    % VP - 20/20
    % VP - 10+10/20
    % VP - 5+5+10/20
    cnnFolder = 'CPM-VP/O3D';
    typeTraining = 'cpm-vp'; % cpm, vp, cpm-vp
    datasets = {'real' 'shapenet'}; % 'real', 'shapenet', 'synthetic'
    allClasses = true;
    numStagesCPM = 6;
    sizePatch = 224;
    sizeBatchTrain = 24;
    sizeBatchTest = 6;
    maxIterCNN = 200000;
    stepSize = 150000;
    lr_cnn = 0.00005;
    minSize = 0;
    keepAR = true;
    padding = 8;
    typeVP = 'class'; % class, reg, map
    onlyAZ = false;
    binSize = 1.0;
    typeData = takeFirstChars(datasets, 2);
    if(~exist('nameType','var'))
        if(~onlyAZ)
            str_onlyAZ = 'all';
        else
            str_onlyAZ = 'az';
        end
        nameType = typeTraining;
        if(strfind(typeTraining, 'vp'))
           nameType = [nameType '_' typeVP '_' str_onlyAZ '_' typeData];
        elseif(strcmp(typeTraining, 'cpm'))
            nameType = [nameType '_' typeData]; 
        end
        if(minSize > 0)
           nameType = [nameType '_' num2str(minSize)];
       end
    end
    
    fprintf('\nMulti-Object/Pose estimation:\n');
    fprintf('Dataset: %s\n', class(input.targetDataset));
    
    [input, data, ~, valData] = getData(input, 'target');
    
    srcData = input.sourceDataset;
    tgtData = input.targetDataset;
    strExtra = ['_' num2str(minSize)];
    if(isprop(tgtData,'addKps'))
        if(tgtData.addKps)
            strExtra = [strExtra '_kps'];
        end
    end
    if(isprop(tgtData,'addImageNet3D'))
        if(tgtData.addImageNet3D)
            strExtra = [strExtra '_i3d'];
        end
    end
    if(length(input.targetDataset.classes) > 12)
        strExtra = [strExtra '_' num2str(length(input.targetDataset.classes))];
    end
    
    % -> JSON files
    if(ismember('real', datasets))
        isValidation = 0;
        genJSON(input, 'real', data, isValidation, strExtra, minSize, allClasses);
    end
    isValidation = 1;
    genJSON(input, 'real', valData, isValidation, strExtra, minSize, allClasses);
    strExtra = ['_' num2str(minSize)];
    
    % -> Test visualisation
%     f = figure;
%     num = 0;
%     listClasses = unique(data.annotations.classes);
%     for idClass = 8 % 1:length(listClasses)
%         isClass = ismember(data.annotations.classes, listClasses{idClass});
%         c_id = data.annotations.imgId(isClass);
%         c_bb = data.annotations.BB(isClass,:);
%         c_parts = data.annotations.parts(isClass);
%         partLabels = data.partLabels{idClass};
%         fprintf('[%s - id %d] number of training samples: %d\n', listClasses{idClass}, idClass, length(c_id));
%         for idTest = 1:length(c_id)
%             idTest2 = idTest;
% %             idTest2 = randi(length(c_id),1);
%             img = imread(data.imgPaths{c_id(idTest2)});
%             imshow(img);
%             set(f, 'Position', [2500, 300, 800, 800]);
%             hold on;
%             BB = c_bb(idTest2,:);
%             rectangle('position', [BB(2) BB(1) BB(4) BB(3)], 'LineWidth', 1, 'EdgeColor', [0.9 0.9 0.2]);
%             parts = c_parts{idTest2};
%             for i = 1:size(parts,1)
%                 if(parts(i,3) > 0)
%                     scatter(parts(i,2), parts(i,1), 40, 'MarkerFaceColor', [0.9 0.9 0.2], 'MarkerEdgeColor',[0.9 0.9 0.0], 'LineWidth', 0.1);
%                     if((parts(i,1) ~= 0 && parts(i,2) ~= 0))
%                         text(parts(i,2)+1, parts(i,1)+1, partLabels{i}, 'HorizontalAlignment', 'left', 'Color', [0.9 0.9 0.2], 'FontSize', 10);
%                     end
%                 end
%             end
%             hold off;
%             keyboard;
%         end
%     end
%     keyboard;
    
    % -> Generate synthetic data (uncomment when needed)
    if(ismember('synthetic', datasets))
        input.sourceDataset = Synthetic;
        [~, synData] = getData(input, 'source');
        % synData = [];
        isValidation = 0;
        genJSON(input, 'synthetic', synData, isValidation, strExtra, minSize, allClasses);
    end
    % -> Generate shapenet data (uncomment when needed)
    if(ismember('shapenet', datasets))
        input.sourceDataset = ShapeNet;
        srcData = input.sourceDataset;
        % [~, shapeData] = getData(input, 'source');
        shapenetData = [];
        folderPath = [input.PATH_DATA srcData.path class(input.targetDataset) '\json_' num2str(minSize) '\'];
        isValidation = 0;
        % genJSON_VP(input, shapeData, isValidation, minSize, allClasses, folderPath);
    end
%     keyboard;
    
    % -> LMDB files for real dataset
    for idxData = 1:length(datasets)
        pathJSON = [input.PATH_DATA tgtData.path];
        if(strcmpi(datasets{idxData},'synthetic'))
            input.sourceDataset = Synthetic;
            pathJSON = [input.PATH_DATA input.sourceDataset.path input.sourceDataset.sub_path]; 
        end
        if(ismember(datasets{idxData}, {'real', 'synthetic'}))
            if(allClasses)
                % Training data
                if(~exist([pathJSON 'lmdb' strExtra '\all'], 'dir'))
                    createDir([pathJSON 'lmdb' strExtra '\all']);
                    system(['py pose/CPM/genLMDB.py ' pathJSON 'json' strExtra '\all' ...
                            '_annotations.json ' pathJSON 'lmdb' strExtra '\all']);
                end
                % Validation data
                if(strcmpi(datasets{idxData},'real') && ~exist([pathJSON 'lmdb' strExtra '\all_val'], 'dir'))
                    createDir([pathJSON 'lmdb' strExtra '\all_val']);
                    system(['py pose/CPM/genLMDB.py ' pathJSON 'json' strExtra '\all_val' ...
                            '_annotations.json ' pathJSON 'lmdb' strExtra '\all_val']);
                end
            else
                for iCl = 1:length(tgtData.classes)
                    % Training data
                    if(~exist([pathJSON 'lmdb' strExtra '\' tgtData.classes{iCl}], 'dir'))
                        createDir([pathJSON 'lmdb' strExtra '\' tgtData.classes{iCl}]);
                        system(['py pose/CPM/genLMDB.py ' pathJSON 'json' strExtra '\' tgtData.classes{iCl} ...
                            '_annotations.json ' pathJSON 'lmdb' strExtra '\' tgtData.classes{iCl}]);
                    end
                    % Validation data
                    if(strcmpi(datasets{idxData},'real') && ~exist([pathJSON 'lmdb' strExtra '\' tgtData.classes{iCl} '_val'], 'dir'))
                        createDir([pathJSON 'lmdb' strExtra '\' tgtData.classes{iCl} '_val']);
                        system(['py pose/CPM/genLMDB.py ' pathJSON 'json' strExtra '\' tgtData.classes{iCl} '_val' ...
                            '_annotations.json ' pathJSON 'lmdb' strExtra '\' tgtData.classes{iCl} '_val']);
                    end
                end
            end
        elseif(strcmpi(datasets{idxData},'shapenet'))
            % nameDataset = class(input.targetDataset);
            input.sourceDataset = ShapeNet;
            srcData = input.sourceDataset;
            nameDataset = class(input.targetDataset);
            % -> Create LMDB ShapeNet for given real dataset
            if(allClasses)
                if(~exist([input.PATH_DATA srcData.path nameDataset '\lmdb_' num2str(minSize) '\all'], 'dir'))
                    createDir([input.PATH_DATA srcData.path nameDataset '\lmdb_' num2str(minSize) '\all']);
                    system(['py pose/CPM/genLMDB_vp.py ' input.PATH_DATA srcData.path nameDataset '\json_' num2str(minSize) '\all' ...
                        '_annotations.json ' input.PATH_DATA srcData.path nameDataset '\lmdb_' num2str(minSize) '\all']);
                end
            else
                for iCl = 1:length(tgtData.classes)
                    if(~exist([input.PATH_DATA srcData.path nameDataset '\lmdb_' num2str(minSize) '\' tgtData.classes{iCl}], 'dir'))
                        createDir([input.PATH_DATA srcData.path nameDataset '\lmdb_' num2str(minSize) '\' tgtData.classes{iCl}]);
                        system(['py pose/CPM/genLMDB_vp.py ' input.PATH_DATA srcData.path nameDataset '\json_' num2str(minSize) '\' tgtData.classes{iCl} ...
                            '_annotations.json ' input.PATH_DATA srcData.path nameDataset '\lmdb_' num2str(minSize) '\' tgtData.classes{iCl}]);
                    end
                end
            end
        end
    end
%     keyboard;

    % -> Create Prototxt
    folder = genProtoFile(input, cnnFolder, nameType, typeTraining, srcData, tgtData, sizePatch, ...
        sizeBatchTrain, sizeBatchTest, maxIterCNN, stepSize, lr_cnn, numStagesCPM, datasets, typeVP, ...
        onlyAZ, strExtra, minSize, binSize, allClasses, keepAR, padding);

    % -> Lunch CNN training
    path_script = [folder '/train.sh'];
    fid = fopen(path_script, 'w');
    weights = '';
    pathCaffe = [strrep(input.PATH_CAFFE,'\','/') 'VS2013/bin/Release/caffe'];
    pathCNN = strrep(input.PATH_CNN, '\', '/');
    if(~allClasses)
        for idx = 1:length(tgtData.classes)
            c = tgtData.classes{idx};
            fprintf(['Training object class: ' c '\n']);
            if(strcmpi(typeTraining,'vp'))
                weights = [' --weights=' pathCNN 'VGG-16/VGG-16.caffemodel'];
            elseif(strcmpi(typeTraining,'cpm-vp'))
                weights = [' --weights=' pathWeights];
            end
            line_script = [pathCaffe ' train --solver=' folder '/' c '/' c '_solver.prototxt' weights ' --gpu=' num2str(gpuID)];
            fprintf(fid,'%s\n',[line_script ' 2>&1 | tee ' folder '/' c '/' c '_log']);
        end
    else % only one call for all classes
        if(strcmpi(typeTraining,'vp'))
            weights = [' --weights=' pathCNN 'VGG-16/VGG-16.caffemodel'];
        elseif(strfind(typeTraining,'cpm'))
            weights = [' --weights=' pathWeights];
        end
        line_script = [pathCaffe ' train --solver=' folder '/all/all_solver.prototxt' weights ' --gpu=' num2str(gpuID)];
        fprintf(fid,'%s\n',[line_script ' 2>&1 | tee ' folder '/all/all_log']);
    end
    fclose(fid);
end

function folder = genProtoFile(input, cnnFolder, nameType, typeTraining, srcData, data, sizePatch, ...
    sizeBatchTrain, sizeBatchTest, maxIterCNN, stepSize, lr_cnn, numStagesCPM, nameDatasets, typeVP, ...
    onlyAZ, strExtra, minSizeSyn, binSize, allClasses, keepAR, padding)
    
    pathCNN = strrep(input.PATH_CNN, '\', '/');
    pathVal = strrep([input.PATH_DATA data.path], '\', '/');
    if(ismember('real', nameDatasets))
        pathData = strrep([input.PATH_DATA data.path], '\', '/');
    else
        pathData = [];
    end
    if(ismember('synthetic', nameDatasets))
        srcData = Synthetic;
        synPathData = strrep([input.PATH_DATA srcData.path srcData.sub_path '\'], '\', '/');
    else 
        synPathData = [];
    end
    if(ismember('shapenet', nameDatasets))
        srcData = ShapeNet;
        shapePathData = strrep([input.PATH_DATA srcData.path class(input.targetDataset) '\'], '\', '/');
    else
        shapePathData = [];
    end
    sizeBatchTrain = num2str(sizeBatchTrain);
    sizeBatchTest = num2str(sizeBatchTest);
    maxIterCNN = num2str(maxIterCNN);
    stepSize = num2str(stepSize);
    lr_cnn = num2str(lr_cnn);
    numStagesCPM = num2str(numStagesCPM);
    sizePatch = num2str(sizePatch);
    binSize = num2str(binSize);
    minSizeSyn = num2str(minSizeSyn);
    onlyAZ = num2str(onlyAZ);
    keepAR = num2str(keepAR);
    padding = num2str(padding);
    numClasses = num2str(length(data.classes));    
    
    % Path of CNN training files
    folder = [pathCNN cnnFolder '/' nameType];
    if(allClasses)
        pathVal = [pathVal 'lmdb' strExtra '/all_val'];
        if(~isempty(pathData))
            pathData = [pathData 'lmdb' strExtra '/all'];
        else
            pathData = '""';
        end
        if(~isempty(synPathData))
            synPathData = [synPathData 'lmdb_' minSizeSyn '/all'];
        else
            synPathData = '""';
        end
        if(~isempty(shapePathData))
            shapePathData = [shapePathData 'lmdb_' minSizeSyn '/all'];
        else
            shapePathData = '""';
        end
        if(strcmpi(typeTraining, 'vp'))
            system(['py pose/CPM/genProto_vp_all.py ' folder ' '  pathData ' ' pathVal ' ' synPathData ' ' ...
                shapePathData ' ' sizePatch ' ' sizeBatchTrain  ' ' sizeBatchTest ' ' maxIterCNN ' ' lr_cnn ' ' ...
                stepSize ' '  typeVP ' ' onlyAZ, ' ' binSize ' '  keepAR ' ' padding ' ' numClasses]);
        elseif(strfind(typeTraining, 'cpm')) % + cpm-vp
            if(strcmpi(typeTraining,'cpm'))
                typeVP = '""'; binSize = num2str(0);
            end
            maxLengthParts = num2str(max(cellfun('length', data.parts)));
            sumLengthParts = num2str(sum(cellfun('length', data.parts)));
            system(['py pose/CPM/genProto_cpmVGG.py all ' folder ' ' pathData ' ' pathVal ' ' synPathData ' ' shapePathData ' ' ...
                sizePatch ' ' sizeBatchTrain ' ' sizeBatchTest ' ' maxIterCNN ' ' lr_cnn ' ' stepSize ' ' numClasses ' ' ...
                maxLengthParts ' ' sumLengthParts ' '  numStagesCPM ' ' keepAR ' ' padding ' ' typeVP ' ' onlyAZ ' ' binSize]);
        end
    else
        numClasses = '1';
        for c = 1:length(data.classes)
            c_name = data.classes{c};
            pathVal_c = [pathVal 'lmdb' strExtra '/' c_name '_val'];
            if(~isempty(pathData))
                pathData_c = [pathData 'lmdb' strExtra '/' c_name];
            else
                pathData_c = '""';
            end
            if(~isempty(synPathData))
                synPathData_c = [synPathData 'lmdb_' minSizeSyn '/' c_name];
            else
                synPathData_c = '""';
            end
            if(~isempty(shapePathData))
                shapePathData_c = [shapePathData 'lmdb_' minSizeSyn '/' c_name];
            else
                shapePathData_c = '""';
            end
            if(strcmpi(typeTraining, 'vp')) % need to be updated! (but not absolutely necessary for experiments)
                system(['py pose/CPM/genProto_vp.py ' c_name ' ' folder ' '  pathData_c ' ' synPathData_c ' ' shapePathData_c ' ' ...
                    sizePatch ' ' sizeBatchTrain ' ' maxIterCNN ' ' typeVP ' '  onlyAZ ' ' binSize ' ' keepAR ' ' padding]);                
                error('function not up to date... please use function for all classes');
            elseif(strfind(typeTraining, 'cpm')) % + cpm-vp
                if(strcmpi(typeTraining,'cpm'))
                    typeVP = ''; binSize = num2str(0);
                end
                system(['py pose/CPM/genProto_cpmVGG.py ' c_name ' ' folder ' ' pathData_c ' ' pathVal_c ' ' synPathData_c ' ' shapePathData_c ' ' ...
                    sizePatch ' ' sizeBatchTrain ' ' sizeBatchTest ' ' maxIterCNN ' ' lr_cnn ' ' stepSize ' ' numClasses ' ' ...
                    num2str(length(data.parts{c})) ' '  numStagesCPM ' ' keepAR ' ' padding ' ' typeVP ' ' onlyAZ ' ' binSize]);
            end
        end
    end
end

function jointNames = takeFirstChars(listNames, numChars)

    jointNames = cellfun(@(x) x(1:numChars), listNames, 'UniformOutput', false);
    jointNames = cellfun(@(x) [upper(x(1)) lower(x(2:end))], jointNames, 'UniformOutput', false);
    jointNames = strjoin(jointNames,'');
    
end
