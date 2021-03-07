function genJSON_VP(input, data, isValidation, minPxl, allClasses, folderPath)

    dataset = input.sourceDataset;
    createDir(folderPath);
    
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
    for idxClass = 1:length(dataset.classes)
        
        if(~allClasses)
            if(exist([folderPath dataset.classes{idxClass} strVal '_annotations.json'], 'file'))
                continue;
            end
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
            vp = zeros(length(idxSameImg),3);
            for idxObjImg = 1:length(idxSameImg)
                id = isClassId(idxObj + idxObjImg - 1);
                BB = data.annotations.BB(id,:); 
                vp(idxObjImg,:) = [data.annotations.vp.azimuth(id), ...
                    data.annotations.vp.elevation(id), data.annotations.vp.plane(id)];
                bb2D(idxObjImg,:) = [BB(2) BB(1) BB(2)+BB(4)-1 BB(1)+BB(3)-1];
                pos2D(idxObjImg,:) = [BB(2) + BB(4)/2, BB(1) + BB(3)/2];
            end
            listIdxObjs = (1:length(idxSameImg));
            for idxObjImg = 1:length(idxSameImg)
                
                if(BB(3) < minPxl || BB(4) < minPxl)
                    idxObj = idxObj + 1;
                    continue;
                end
                
                % -> (0) Reset metadata and assign dataset name
                metadata(idxMeta).dataset = 'ShapeNet';

                % -> (1) Validation (80/20)
                metadata(idxMeta).isValidation = isValidation;
                
                % -> (2) Path image
                metadata(idxMeta).img_paths = data.imgPaths{data.annotations.imgId(isClassId(idxObj))};

                % -> (3) Store resolution: width and height
                metadata(idxMeta).img_width = size(img,2);
                metadata(idxMeta).img_height = size(img,1);
                
                % -> (4) Object class id
                metadata(idxMeta).objId = idxClass;
                
                % -> (5) Index annotation for this class and num in image
                metadata(idxMeta).annolist_index = idxMeta;
                
                % -> (6) Centres position of obj in image
                metadata(idxMeta).objpos = pos2D(idxObjImg,:);
                % -> (6.1) BB of obj in image
                metadata(idxMeta).bb = bb2D(idxObjImg,:);
                
                % -> (7) Viewpoint information (az, el, pl)
                metadata(idxMeta).vp = vp(idxObjImg,:);
                
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
