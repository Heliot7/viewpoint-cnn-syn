function [data, testData] = getPascalKps(input, dataset)

    trunc = zeros(12,2);
    occ = zeros(12,2);

    data = defineData();
    testData = defineData();
    phases = {'train', 'test'};
    dirXML = [input.PATH_DATA 'Real/Multi/PASCAL_VOC12/train/Annotations/'];
    dirAnno = [input.PATH_DATA 'Real/Multi/PASCAL_VOC12/train/ImageSets/Main/'];
    dirImg = [input.PATH_DATA 'Real/Multi/PASCAL_VOC12/train/JPEGImages/'];
    file = fopen([dirAnno 'train.txt']);
    fileData = textscan(file, '%s');
    fclose(file);
    trainIds = sort_nat(fileData{1});
    
    file = fopen([dirAnno 'val.txt']);
    fileData = textscan(file, '%s');
    fclose(file);
    testIds = sort_nat(fileData{1});
    
    % 3165 (train) 3173 (val)
    idxs = [1 1];
    for c = 1:length(dataset.classes)
        class = dataset.classes{c};        
        load(['segkps\' class]);
        numParts = size(keypoints.labels,1);
        for p = 1:length(phases)
            ids = eval([phases{p} 'Ids']);
            auxData = defineData(); 
            auxData.partLabels = {keypoints.labels'};
            for a = 1:length(ids)
                % Remove if annotation not in train/val ids
                id_key = find(ismember(keypoints.voc_image_id, ids(a)));
                if(isempty(id_key))
                    continue;
                end                
                rec = VOCreadxml([dirXML ids{a} '.xml']);
                N = length(rec.annotation.object);
                for n_id = 1:length(id_key)
                    for n = 1:N
                        bbKey = keypoints.bbox(id_key(n_id),:);
                        obj = rec.annotation.object(n);
                        bb = obj.bndbox;
                        if(~strcmpi(obj.name,class))
                            continue;
                        end
                        if(bbKey(1) ~= str2double(bb.xmin) || bbKey(2) ~= str2double(bb.ymin))
                            continue;
                        end
                        if(~strcmpi(obj.difficult,'0'))
                            continue;
                        end
                        if(~strcmpi(obj.occluded,'0'))
                            occ(c,p) = occ(c,p) + 1;
                        end
                        % take only truncated that are "not" occluded
                        if(~strcmpi(obj.truncated,'0') && strcmpi(obj.occluded,'0'))
                            trunc(c,p) = trunc(c,p) + 1;
                        end
                        % Get basic info
                        auxData.imgPaths = [auxData.imgPaths; {[dirImg ids{a} '.jpg']}];
                        auxData.annotations.imgId = [auxData.annotations.imgId; idxs(p)]; 
                        auxData.annotations.classes = [auxData.annotations.classes; {class}];
                        idxs(p) = idxs(p) + 1;
                        auxData.annotations.BB = [auxData.annotations.BB; [bbKey(2) bbKey(1) bbKey(4) bbKey(3)]];
                        % Get KPS
                        kps = squeeze(keypoints.coords(id_key(n_id),:,:));
                        objParts = zeros(numParts,3);
                        for idxPart = 1:numParts
                            locPart = kps(idxPart,2:-1:1);
                            if(~isnan(locPart(1)))
                                objParts(idxPart,:) = [locPart, 1];
                            end
                        end
                        auxData.annotations.parts = [auxData.annotations.parts; {objParts}];
                        % Get VPS
                        imgAnnotation = load([input.PATH_DATA 'Real/Multi/PASCAL3D/Annotations/' class '_pascal/' ids{a}]);
                        objAnno = imgAnnotation.record.objects(n);
                        if(objAnno.viewpoint.distance == 0)
                            azimuth = objAnno.viewpoint.azimuth_coarse;
                        else
                            if(~isfield(objAnno.viewpoint,'azimuth') || isempty(objAnno.viewpoint.azimuth))                            
                                azimuth = objAnno.viewpoint.azimuth_coarse;
                            else
                                azimuth = objAnno.viewpoint.azimuth;
                            end
                        end
                        auxData.annotations.vp.azimuth = [auxData.annotations.vp.azimuth; azimuth];
                        if(objAnno.viewpoint.distance == 0)
                            elevation = objAnno.viewpoint.elevation_coarse;
                        else
                            if(~isfield(objAnno.viewpoint,'elevation') || isempty(objAnno.viewpoint.elevation))
                                elevation = objAnno.viewpoint.elevation_coarse;
                            else
                                elevation = objAnno.viewpoint.elevation;
                            end
                        end
                        auxData.annotations.vp.elevation = [auxData.annotations.vp.elevation; elevation];
                        distance = objAnno.viewpoint.distance;
                        auxData.annotations.vp.distance = [auxData.annotations.vp.distance; distance];
                        plane = objAnno.viewpoint.theta;
                        auxData.annotations.vp.plane = [auxData.annotations.vp.plane; plane];
                        auxData.annotations.camera.focal = [auxData.annotations.camera.focal; objAnno.viewpoint.focal];
                        auxData.annotations.camera.px = [auxData.annotations.camera.px; objAnno.viewpoint.px];
                        auxData.annotations.camera.py = [auxData.annotations.camera.py; objAnno.viewpoint.py];
                        auxData.annotations.camera.viewport = [auxData.annotations.camera.viewport; objAnno.viewpoint.viewport];                    
                        break;
                    end
                end
            end
            % Append info
            if(strcmpi(phases{p},'train'))
                data = appendData(data, auxData);
            else % test
                testData = appendData(testData, auxData);
            end
        end
    end
    
%     for i = 1:length(dataset.classes)
%         fprintf('[train] %s: %d samples (occ: %d trunc: %d)\n', dataset.classes{i}, sum(ismember(data.annotations.classes,dataset.classes(i))), ...
%             occ(i,1), trunc(i,1));
%         fprintf('[test] %s: %d samples (occ: %d trunc: %d)\n', dataset.classes{i}, sum(ismember(testData.annotations.classes,dataset.classes(i))), ...
%             occ(i,2), trunc(i,2));
%     end
    
%     for i = 1:length(dataset.classes)
%         fprintf('[train] %s: %d rejected, %d saved\n', dataset.classes{i}, numRejectedKps_train(i), total_train(i));
%         fprintf('[test] %s: %d rejected, %d saved\n', dataset.classes{i}, numRejectedKps_test(i), total_test(i));
%     end
    
%     % Info files:
%     for i = 1:length(dataset.classes)
%         class = dataset.classes{i};
%         idxs = ismember(data.annotations.classes, class);
%         bb = data.annotations.BB(idxs,:);
%         num = size(bb,1);
%         row = [mean(bb(:,3)) std(bb(:,3))];
%         col = [mean(bb(:,4)) std(bb(:,4))];
%         fprintf('[%s]\nTRAIN %d samples, row: %.1f,%.1f col: %.1f,%.1f\n', class, num, row, col);
%         idxs = ismember(testData.annotations.classes, class);
%         bb = testData.annotations.BB(idxs,:);
%         num = size(bb,1);
%         row = [mean(bb(:,3)) std(bb(:,3))];
%         col = [mean(bb(:,4)) std(bb(:,4))];
%         fprintf('TEST %d samples, row: %.1f,%.1f col: %.1f,%.1f\n', num, row, col);
%     end
% %     keyboard;

    % Check how many are 0,0,0 in VP:
%     fprintf('\n')
%     % Training:
%     noAZ = (data.annotations.vp.azimuth == 0);
%     noEL = (data.annotations.vp.elevation == 0);
%     noPL = (data.annotations.vp.plane == 0);
%     noTOTAL = noAZ + noEL + noPL;
%     noOBJ = find(noTOTAL == 3);
%     fprintf('TRAIN 0,0,0: %d\n', length(noOBJ));
%     % Test:
%     noAZ = (testData.annotations.vp.azimuth == 0);
%     noEL = (testData.annotations.vp.elevation == 0);
%     noPL = (testData.annotations.vp.plane == 0);
%     noTOTAL = noAZ + noEL + noPL;
%     noOBJ = find(noTOTAL == 3);
%     fprintf('TEST 0,0,0: %d\n', length(noOBJ));
%     keyboard;

end

function data = defineData()

    data.imgPaths = []; data.annotations.imgId = []; data.annotations.BB = [];
    data.annotations.classes = []; data.annotations.parts = [];
    data.annotations.vp.azimuth = []; data.annotations.vp.elevation = [];
    data.annotations.vp.distance = []; data.annotations.vp.plane = [];
    data.annotations.camera.px = []; data.annotations.camera.py = [];
    data.annotations.camera.focal = []; data.annotations.camera.viewport = []; 
    data.partLabels = [];

end

function data = appendData(data, auxData)

    data.imgPaths = [data.imgPaths; auxData.imgPaths];
    data.annotations.imgId = [data.annotations.imgId; auxData.annotations.imgId];
    data.annotations.classes = [data.annotations.classes; auxData.annotations.classes];
    data.annotations.parts = [data.annotations.parts; auxData.annotations.parts];
    data.annotations.BB = [data.annotations.BB; auxData.annotations.BB];
    data.annotations.vp.azimuth = [data.annotations.vp.azimuth; auxData.annotations.vp.azimuth];
    data.annotations.vp.elevation = [data.annotations.vp.elevation; auxData.annotations.vp.elevation];
    data.annotations.vp.distance = [data.annotations.vp.distance; auxData.annotations.vp.distance];
    data.annotations.vp.plane = [data.annotations.vp.plane; auxData.annotations.vp.plane];
    data.annotations.camera.px = [data.annotations.camera.px; auxData.annotations.camera.px];
    data.annotations.camera.py = [data.annotations.camera.py; auxData.annotations.camera.py];
    data.annotations.camera.focal = [data.annotations.camera.focal; auxData.annotations.camera.focal];
    data.annotations.camera.viewport = [data.annotations.camera.viewport; auxData.annotations.camera.viewport]; 
    data.partLabels = [data.partLabels; auxData.partLabels];
    
end
