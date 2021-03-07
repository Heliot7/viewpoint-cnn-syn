function genJSON(input, typeDataset, data, isValidation, strExtra, minPxl, allClasses)

    if(strcmpi(typeDataset,'real'))
        dataset = input.targetDataset;
        folderPath = [input.PATH_DATA dataset.path 'json' strExtra '\'];
    else % synthetic
        dataset = input.sourceDataset;
        folderPath = [input.PATH_DATA dataset.path dataset.sub_path 'json' strExtra '\'];
    end
    
    createDir(folderPath);
    
    % Store all names of all limbs to know them:
    fileId = fopen([folderPath 'parts.txt'], 'w');
    for idxClass = 1:length(dataset.classes)
        fprintf(fileId,'%s:\n======\n', dataset.classes{idxClass});
        parts = dataset.parts{idxClass};
        for idxPart = 1:length(parts)
            fprintf(fileId,'%s\n', parts{idxPart});
        end
        fprintf(fileId,'\n');
    end
    fclose(fileId);
    
    strVal = '';
    if(isValidation)
        strVal = '_val';
    end
    
    if(allClasses)
        if(exist([folderPath 'all' strVal '_annotations.json'], 'file'))
            return;
        end
    end
    
    % Store JSON info for all classes!
    idxMeta = 1; metadata = [];
    dataset_name = class(dataset);
    if(length(dataset.classes) > 12)
        dataset_name = [dataset_name '_' num2str(length(dataset.classes))];
    end
    if(isprop(dataset,'addKps'))
        if(dataset.addKps)
            dataset_name = [dataset_name '_kps'];
        end
    end
    for idxClass = 1:length(dataset.classes)
        
        if(~allClasses)
            if(exist([folderPath dataset.classes{idxClass} strVal '_annotations.json'], 'file'))
                continue;
            end
        end
        
        % Take part relation metadata (left <-> right)
        str_parts = dataset.parts{idxClass};
        rel_parts = findKpsPerm(str_parts);
        
        % Select joint position for current class and num joints
        num_parts = length(str_parts);
        if(allClasses)
            pos_parts = length(cat(2,input.targetDataset.parts{1:idxClass-1})) + 1;
        else
            pos_parts = 1;
        end
        
        if(~allClasses)
            idxMeta = 1;
            metadata = [];
        end
        idxObj = 1;
        isClass = ismember(data.annotations.classes, dataset.classes{idxClass});
        isClassId = find(isClass);
        while(idxObj <= length(isClassId))
            
            img = imread(data.imgPaths{data.annotations.imgId(isClassId(idxObj))});
            
            % Iterate over instance of same image
            idxSameImg = find(isClass & ismember(data.annotations.imgId, data.annotations.imgId(isClassId(idxObj))));
            % Pre: compute/store 2D positions
            pos2D = zeros(length(idxSameImg),2);
            bb2D = zeros(length(idxSameImg),4);
            parts2D = cell(1,length(idxSameImg));
            scales = zeros(1, length(idxSameImg));
            vp = zeros(length(idxSameImg),3);
            for idxObjImg = 1:length(idxSameImg)
                id = isClassId(idxObj + idxObjImg - 1);
                BB = data.annotations.BB(id,:); 
                vp(idxObjImg,:) = [data.annotations.vp.azimuth(id), ...
                    data.annotations.vp.elevation(id), data.annotations.vp.plane(id)];
                bb2D(idxObjImg,:) = [BB(2) BB(1) BB(2)+BB(4)-1 BB(1)+BB(3)-1];
                pos2D(idxObjImg,:) = [BB(2) + BB(4)/2, BB(1) + BB(3)/2];
                scales(idxObjImg) = max(BB(3:4)) / 200.0;
                part = data.annotations.parts{id,:};
                parts2D{idxObjImg} = [part(:,[2,1,3]), rel_parts];
            end
            listIdxObjs = (1:length(idxSameImg));
            for idxObjImg = 1:length(idxSameImg)
                
                if(BB(3) < minPxl || BB(4) < minPxl)
                    idxObj = idxObj + 1;
                    continue;
                end
                
                % -> (0) Reset metadata and assign dataset name
                metadata(idxMeta).dataset = dataset_name;

                % -> (1) Validation (80/20)
                % if(idxObj < 0.8*length(isClassId))
                metadata(idxMeta).isValidation = isValidation;
                % else
                    % metadata(idxMeta).isValidation = 1;
                % end

                % -> (2) Path image
                metadata(idxMeta).img_paths = data.imgPaths{data.annotations.imgId(isClassId(idxObj))};

                % -> (3) Store resolution: width and height
                metadata(idxMeta).img_width = size(img,2);
                metadata(idxMeta).img_height = size(img,1);
                
                % -> (4) Object class id
                metadata(idxMeta).objId = idxClass;
                
                % -> (5) Index annotation for this class and num in image
                metadata(idxMeta).annolist_index = idxMeta;
                metadata(idxMeta).people_index = idxObjImg;
                
                % -> (6) Centres position of obj in image
                metadata(idxMeta).objpos = pos2D(idxObjImg,:);
                % -> (6.1) BB of obj in image
                metadata(idxMeta).bb = bb2D(idxObjImg,:);
                
                % -> (7) Scale of instance
                metadata(idxMeta).scale_provided = scales(idxObjImg);
                
                % -> (8) 2D position and occluded or not for each part
                metadata(idxMeta).num_parts = num_parts;
                metadata(idxMeta).pos_parts = pos_parts;
                metadata(idxMeta).joint_self = parts2D{idxObjImg};

                % -> (9) Viewpoint information (az, el, pl)
                metadata(idxMeta).vp = vp(idxObjImg,:);
                
                % -> (10) Other instances in same image
                metadata(idxMeta).numOtherPeople = length(idxSameImg) - 1;
                if(length(idxSameImg) > 1)
                    metadata(idxMeta).objpos_other = pos2D(listIdxObjs ~= idxObjImg,:);
                    metadata(idxMeta).scale_provided_other = scales(listIdxObjs ~= idxObjImg);
                    metadata(idxMeta).joint_others = parts2D(listIdxObjs ~= idxObjImg);
                    metadata(idxMeta).vp_others = vp(listIdxObjs ~= idxObjImg,:);
                end
                
                fprintf('Class %s, instance %d/%d\n', dataset.classes{idxClass}, idxObj, length(isClassId));
                
                % Update idxObj for next dataset instance
                idxObj = idxObj + 1;
                idxMeta = idxMeta + 1;
                
            end

        end
        
        % Save file
        if(~allClasses)
            opt.FloatFormat = '%.3f';
            opt.FileName =  [folderPath dataset.classes{idxClass} strVal '_annotations.json'];
            savejson('root', metadata, opt);
        end
        
    end
    
    if(allClasses)
        % Save file
        opt.FloatFormat = '%.3f';
        opt.FileName =  [folderPath 'all' strVal '_annotations.json'];
        savejson('root', metadata, opt);
    end

end
