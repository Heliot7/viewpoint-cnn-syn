function createTransfer_LMDB()

    input = InputParameters;
    
    % -> Comment if gt:
    isGT = false;
    % -> Info label transfer:
    input.targetDataset = Pascal3D;
    input.sourceDataset = Synthetic;
    input.sourceDataset.sub_path = 'Output_1d\';
    input.sourceDataset.classes = input.targetDataset.classes;
    input.targetDataset.azimuth = 24;
    input.typeDescriptor = 'CNN-pool5';
    input.cnnName = 'VGG-16';
    input.cnnModel = 'VGG-16';
    input.isDA = true;

    % -> Get data
    [input, data, ~, testData] = getData(input, 'target');
    
    % -> Update VPs with transferred data
    if(~isGT)
        for c = 1:length(input.targetDataset.classes)
            nameClass = input.targetDataset.classes{c};
            nameTransfer = nameTransferLabel(input, nameClass);
            load([input.PATH_DATA input.targetDataset.path 'transferLabel\' nameTransfer]);
            metaData = transferLabel.labels;
            azimuth = transferLabel.values(:,ismember(metaData,'azimuth'));
            azimuth = cell2mat(cellfun(@str2num, azimuth, 'UniformOutput', false));
            isClass = ismember(data.annotations.classes, nameClass);
            data.annotations.vp.azimuth(isClass) = azimuth;
        end
        if(c > 1)
            nameTransfer =  nameTransferLabel(input, 'all');
        end
    else
        nameTransfer = 'gt';
    end
    % -> Add elevation and in-plane if do not exist    
    if(~isfield(data.annotations.vp,'elevation'))
        data.annotations.vp.elevation = zeros(length(data.annotations.vp.azimuth),1);
        if(~isempty(testData))
            testData.annotations.vp.elevation = zeros(length(testData.annotations.vp.azimuth),1);
        end
    end
    if(~isfield(data.annotations.vp,'plane'))
        data.annotations.vp.plane = zeros(length(data.annotations.vp.azimuth),1);
        if(~isempty(testData))
            testData.annotations.vp.plane = zeros(length(testData.annotations.vp.azimuth),1);
        end
    end
    
    % -> JSON files
    allClasses = true;
    isValidation = 0;
    minSize = 0;
    folderPath = [input.PATH_DATA input.targetDataset.path];
    folderJSON = [folderPath 'json\' nameTransfer '\'];
    folderLMDB = [folderPath 'lmdb\' nameTransfer '\'];
    genJSON_VP(input, data, isValidation, minSize, allClasses, folderJSON);
    if(isGT && ~isempty(testData))
        isValidation = 1;
        genJSON_VP(input, testData, isValidation, minSize, allClasses, folderJSON);
    end
    if(~exist(folderLMDB, 'dir'))
        createDir(folderLMDB);
        system(['py pose/CPM/genLMDB_vp.py ' folderJSON 'all_annotations.json ' folderLMDB 'all']);
        if(isGT && ~isempty(testData))
            system(['py pose/CPM/genLMDB_vp.py ' folderJSON 'all_val_annotations.json ' folderLMDB 'all_val']);
        end
    end
    
end

