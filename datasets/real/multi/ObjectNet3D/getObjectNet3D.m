function [data, testData] = getObjectNet3D(input, dataset, nameDataset)

    big12 = {'aeroplane', 'bicycle', 'boat', 'bottle', ...
        'bus', 'car', 'chair', 'diningtable', 'motorbike', ...
        'sofa', 'train', 'tvmonitor'};
    
    % figure;
    % -> Special cases where motorbike handles are swapped (train-val)
    % swapMbikeTrain = [13360 13365 13367 13373 13377 13383 13391 13392 13393 13394 13395 13396 13397 13400 13405 13406 13407 13410 13412 13418 13420 13421 13425 13429 13435 13438 13440 13442 13444 13445 13447 13448 13449 13450 13452 13453 13454 13456 13457 13459 13460 13462 13466 13472 13473 13476 13477 13480 13480 13481 13482 13485 13487 13495 13498 13499 13501 13503 13505 13507 13509 13511 13513 13520 13569 13571 13581 21429 21430 21431 21432 21433 21434 21435 21436 21437 21438 21439 21441 21442 21443 21444 21445 21446 21448 21449 21450 21451 21452 21453 21454 21455 21456 21457 21458 21461 21462 21463 21464 21465 21466 21467 21468 21469 21470 21472 21473 21474 21477 21479 21482 21483 21484 21485 21486 21490 21492 21494 21495 21497 21498 21499 21500 21501 21502 21503 21504 21505];
    % swapMbikeTest = [13214 13218 13220 13225 13226 13228 13229 13231 13236 13238 13239 13240 13241 13242 13243 13246 13247 13249 13253 13255 13256 13257 13259 13260 13261 13266 13268 13269 13270 13271 13275 13276 13280 13284 13285 13287 13289 13290 13291 13292 13300 13304 13305 13309 13311 13312 13314 13315 13316 13319 13321 13324 13326 13332 13338 13339 13341 13344 13368 13398 13398 13408 13418 21068 21070 21071 21074 21076 21077 21078 21081 21082 21083 21084 21085 21086 21087 21088 21090 21091 21092 21093 21094 21095 21097 21098 21099 21100 21101 21102 21103 21104 21105 21107 21109 21111 21112 21114 21115 21116 21118 21119 21120 21124 21125 21126 21128 21130 21132 21133 21134 21137 21138 21139 21140 21141 21142 21144 21145 21146 21148 21149 21151 21153 21154 21155 21157 21158 21161 21162 21163];
    % -> Special cases where motorbike handles are swapped (train+val-test)
    swapMbikeTrain = [26579,26591,26594,26605,26609,26619,26639,26640,26643,26644,26646,26650,26652,26657,26667,26668,26669,26674,26676,26684,26687,26688,26694,26703,26713,26721,26725,26727,26734,26741,26745,26747,26748,26750,26755,26756,26759,26763,26764,26768,26770,26775,26783,26794,26795,26801,26803,26807,26808,26809,26812,26815,26831,26836,26837,26839,26841,26845,26852,26854,26856,26859,26874,26975,26977,26995,42510,42512,42513,42516,42519,42522,42527,42529,42530,42531,42532,42534,42537,42539,42540,42542,42547,42551,42555,42556,42559,42561,42562,42563,42564,42568,42569,42571,42574,42576,42577,42578,42579,42581,42583,42586,42588,42589,42594,42601,42606,42612,42614,42618,42620,42621,42627,42629,42639,42643,42646,42648,42652,42653,42655,42656,42658,42660,42662,42666,42667,26571,26576,26581,26589,26590,26595,26597,26599,26616,26625,26626,26628,26629,26630,26631,26636,26637,26641,26648,26651,26653,26655,26660,26664,26665,26682,26691,26692,26695,26696,26704,26705,26715,26722,26723,26729,26732,26733,26735,26736,26749,26757,26758,26766,26771,26773,26776,26778,26779,26787,26791,26798,26802,26824,26833,26842,26847,26850,26897,26954,26982,27001,42495,42497,42499,42502,42504,42505,42506,42509,42511,42514,42515,42517,42518,42520,42521,42524,42525,42526,42528,42535,42536,42541,42543,42544,42545,42546,42548,42550,42552,42553,42557,42560,42566,42567,42575,42580,42582,42585,42587,42590,42596,42597,42598,42600,42603,42605,42609,42610,42619,42622,42623,42624,42625,42626,42630,42631,42632,42636,42638,42642,42647,42650,42651,42657,42659,42664,42665,42668];
    swapMbikeTest = [26217,26219,26220,26221,26222,26223,26224,26226,26227,26232,26234,26235,26238,26239,26241,26242,26243,26244,26247,26250,26251,26254,26255,26256,26257,26258,26259,26260,26261,26262,26263,26264,26266,26267,26268,26269,26271,26274,26275,26276,26278,26285,26297,26298,26300,26301,26302,26305,26306,26307,26308,26313,26321,26323,26325,26326,26327,26329,26332,26335,26338,26340,26341,26346,26347,26353,26354,26355,26357,26359,26362,26364,26370,26372,26373,26375,26383,26387,26388,26390,26391,26392,26393,26394,26396,26399,26400,26401,26402,26405,26409,26413,26415,26416,26420,26421,26425,26430,26431,26432,26433,26435,26436,26437,26438,26440,26445,26447,26449,26451,26452,26454,26456,26458,26460,26461,26469,26470,26475,26477,26479,26481,26483,26486,26487,26500,26501,26503,26504,26525,26529,26573,26580,26582,26600,26605,26611,26619,26623,26641,41849,41850,41851,41852,41853,41855,41856,41857,41858,41859,41860,41861,41862,41864,41865,41868,41871,41872,41873,41874,41875,41876,41877,41878,41879,41880,41883,41884,41885,41886,41887,41888,41889,41890,41892,41893,41894,41895,41896,41897,41898,41899,41900,41901,41902,41904,41905,41906,41907,41908,41909,41910,41911,41912,41915,41917,41918,41920,4192141922,41923,41924,41925,41926,41927,41930,41931,41932,41934,41935,41936,41937,41938,41939,41940,41941,41942,41943,41945,41946,41948,41949,41950,41951,41952,41953,41954,41955,41956,41957,41958,41959,41960,41961,41963,41964,41966,41967,41968,41969,41970,41971,41972,41973,41974,41977,41978,41979,41980,41981,41982,41983,41984,41986,41987,41989,41990,41991,41992,41993,41994,41995,41996,41997,41999,42000,42001,42002,42003,42006,42007,42009,42010,42011,42012,42013,42014,42015,42017,42018];
    
    numRejectedKps_train = zeros(length(dataset.classes),1);
    numRejectedKps_test = zeros(length(dataset.classes),1);
    total_train = zeros(length(dataset.classes),1);
    total_test = zeros(length(dataset.classes),1);
    PATH = [input.PATH_DATA dataset.path];
    phases = {'train', 'test'};
    strOcc = '_noOcc';
    if(dataset.isOcclusions)
        strOcc = '_occ';
    end
    data = defineData();
    testData = defineData();
    for p = 1:length(phases)
        
%         if(p == 2) % Remove after occ - noOcc tests
%             aux_occ = input.targetDataset.isOcclusions;
%             input.targetDataset.isOcclusions = false;
%             strOcc = '_noOcc';
%         end
        
        cumPos = {};
        
        isFirstPass = true;
        for c = 1:length(dataset.classes)
            
            % In one pass all objectnet data is read
            if(strcmpi(nameDataset,'objectnet') && ~isFirstPass)
                break;
            end
            
            path_mat_data = [PATH 'mat_data\' dataset.classes{c} '_' phases{p} strOcc '.mat'];
            if(~strcmpi(nameDataset,'objectnet'))
                fprintf('(%s) %s - Reading object class (%d) %s\n', phases{p}, nameDataset, c, dataset.classes{c});
            end
            if(exist(path_mat_data,'file'))

                load(path_mat_data);
                
            else

                auxData = defineData();
                class = dataset.classes{c};
                
                % Select list of paths (VOC'12 training set)
                if(strcmpi(phases{p},'train'))
                    if(strcmpi(nameDataset,'pascal'))
                        pathFile = [PATH 'Image_sets\' class '_train.txt'];
                    elseif(strcmpi(nameDataset,'imagenet'))
                        pathFile = [PATH 'Image_sets\' class '_imagenet_train.txt'];
                    elseif(strcmpi(nameDataset,'objectnet'))
                        % pathFile = [PATH 'Image_sets\train.txt'];
                        pathFile = [PATH 'Image_sets\trainval.txt'];
                    end
                else % test (VOC'12 validation set)
                    if(strcmpi(nameDataset,'pascal'))
                        pathFile = [PATH 'Image_sets\' class '_val.txt'];
                    elseif(strcmpi(nameDataset,'imagenet'))
                        pathFile = [PATH 'Image_sets\' class '_imagenet_val.txt'];
                    elseif(strcmpi(nameDataset,'objectnet'))
                        % pathFile = [PATH 'Image_sets\val.txt'];
                        pathFile = [PATH 'Image_sets\test.txt'];
                    end
                end

                file = fopen(pathFile);
                if(strcmpi(nameDataset,'pascal'))
                    fileData = textscan(file, '%s %d');
                elseif(strcmpi(nameDataset,'imagenet') || strcmpi(nameDataset,'objectnet'))
                    fileData = textscan(file, '%s');
                end
                fclose(file);
                allPaths = sort_nat(fileData{1});

                if(strcmpi(nameDataset,'pascal'))
                    idxPos = find(fileData{2} == 1);
                    listAnnotations = strcat(repmat({[PATH 'Annotations\' class '_' nameDataset '\']}, [length(idxPos), 1]), allPaths(idxPos), repmat({'.mat'}, [length(idxPos), 1]));
                elseif(strcmpi(nameDataset,'imagenet'))
                    listAnnotations = strcat(repmat({[PATH 'Annotations\' class '_' nameDataset '\']}, [length(allPaths), 1]), allPaths, repmat({'.mat'}, [length(allPaths), 1]));
                    idxPos = find(ones(length(allPaths),1));
                elseif(strcmpi(nameDataset,'objectnet'))
                    listAnnotations = strcat(repmat({[PATH 'Annotations\']}, [length(allPaths), 1]), allPaths, repmat({'.mat'}, [length(allPaths), 1]));
                    idxPos = find(ones(length(allPaths),1));
                end
                names_class =  allPaths(idxPos);
                cumPos = [cumPos; allPaths(idxPos)];
                if(strcmpi(nameDataset,'pascal'))
                    allMainPaths = repmat({[input.PATH_DATA 'Real/Multi/PASCAL_VOC12/train/JPEGImages/']},[length(allPaths), 1]);
                    allExt = repmat({'.jpg'},[length(allPaths), 1]);
                elseif(strcmpi(nameDataset,'imagenet'))
                    allMainPaths = repmat({[input.PATH_DATA 'Real/Multi/ImageNet3D/Images/' class '_' nameDataset '/']},[length(allPaths), 1]);
                    allExt = repmat({'.jpeg'},[length(allPaths), 1]);                
                elseif(strcmpi(nameDataset,'objectnet'))
                    allMainPaths = repmat({[PATH 'Images/']},[length(allPaths), 1]);
                    allExt = repmat({'.jpeg'},[length(allPaths), 1]);
                end
                auxData.imgPaths = strcat(allMainPaths,allPaths,allExt);
                
                ids = []; annotations = []; classes = {}; parts = {};
                azimuth = []; elevation = []; distance = []; plane = [];
                camera.px = []; camera.py = []; camera.focal = []; camera.viewport = [];
                allParts = cell(1, length(dataset.classes));
                for i = 1:length(listAnnotations)
                    
                    % load new annotations to gather and check if potential sample
                    imgAnnotation = load(listAnnotations{i});
                    objs = imgAnnotation.record.objects;
                    % path = imgAnnotation.record.filename;
                    idx = 0;
                    for anno = 1:length(objs)
                        objAnno = objs(anno);
                        % ... we only consider object class annotations
                        if(~strcmpi(nameDataset,'objectnet') && ~strcmpi(objAnno.class, class))
                            continue;
                        elseif(strcmpi(nameDataset,'objectnet') && ~ismember(objAnno.class, dataset.classes))
                            continue;
                        end
                        % Always ignore difficult objects!
                        if(objAnno.difficult)
                            continue;
                        end
                        % ... we only consider full visible cars
                        if((objAnno.truncated || objAnno.occluded) && ~dataset.isOcclusions)
                           continue;
                        end

                        % -> Save anchors parts (if no anchors, then remove)
                        if(isempty(objAnno.anchors) || ~isfield(objAnno, 'anchors'))
                            if(p == 1)
                                numRejectedKps_train(ismember(dataset.classes,objAnno.class)) = ...
                                    numRejectedKps_train(ismember(dataset.classes,objAnno.class)) + 1;
                            else
                                numRejectedKps_test(ismember(dataset.classes,objAnno.class)) = ...
                                    numRejectedKps_test(ismember(dataset.classes,objAnno.class)) + 1;
                            end
                            continue;
                        else
                            % 1 - visible, 2 - self-occluded, 3 - occluded by other objects
                            % - 4 truncated, 5 - unknown (does not exist)
                            nameParts = fieldnames(objAnno.anchors)';
                            % Remove small amount of backpatck with 
                            if(isNonConsistentSample(objAnno.class, nameParts) || isNonConsistentLabel(objAnno.class, nameParts))
                                if(p == 1)
                                    numRejectedKps_train(ismember(dataset.classes,objAnno.class)) = ...
                                        numRejectedKps_train(ismember(dataset.classes,objAnno.class)) + 1;
                                else
                                    numRejectedKps_test(ismember(dataset.classes,objAnno.class)) = ...
                                        numRejectedKps_test(ismember(dataset.classes,objAnno.class)) + 1;
                                end
                                continue;
                            end
                            if(isempty(nameParts))
                                if(p == 1)
                                    numRejectedKps_train(ismember(dataset.classes,objAnno.class)) = ...
                                        numRejectedKps_train(ismember(dataset.classes,objAnno.class)) + 1;
                                else
                                    numRejectedKps_test(ismember(dataset.classes,objAnno.class)) = ...
                                        numRejectedKps_test(ismember(dataset.classes,objAnno.class)) + 1;
                                end
                                continue;
                            end
                            anchors = objAnno.anchors;
                            objParts = zeros(length(nameParts),3);
                            if(strcmpi(nameDataset,'pascal') || strcmpi(nameDataset,'imagenet'))
                                allParts = {nameParts};
                            else % ObjectNet3D
                                allParts(ismember(dataset.classes, objAnno.class)) = {nameParts};
                            end                           
%                             imshow(imread(auxData.imgPaths{i}));
%                             hold on;
%                             fprintf('%s\n', objAnno.class);
                            for idxPart = 1:length(nameParts)
                                locPart = anchors.(nameParts{idxPart});
                                id_vis = 1;
                                if(~ismember(objAnno.class,big12))
                                    id_vis = 0;
                                end
%                                 printAnchor(nameParts{idxPart}, locPart);
%                                 if(~isempty(locPart.location))
%                                     hold on;
%                                     plot(locPart.location(1), locPart.location(2), ...
%                                         'ro', 'LineWidth', 2, 'MarkerSize', 15, 'Color','m');
%                                     hold off;
%                                 end
                                if(~isempty(locPart.status))
%                                     if(locPart.status > id_vis)
%                                         keyboard;
%                                     end
                                    img_cols = imgAnnotation.record.imgsize(1);
                                    img_rows = imgAnnotation.record.imgsize(2);
                                    if(locPart.status == id_vis)
                                        if(strcmpi(objAnno.class,'motorbike') && strcmpi(nameDataset,'objectnet'))
                                            if((p == 1 && sum(ismember(swapMbikeTrain,i)) > 0 || ...
                                                    p == 2 && sum(ismember(swapMbikeTest,i)) > 0) && ...
                                                    (idxPart == 7 || idxPart == 10))
                                                if(idxPart == 7)
                                                    objParts(10,:) = [locPart.location(2:-1:1), 1];
                                                else % == 10
                                                    objParts(7,:) = [locPart.location(2:-1:1), 1];
                                                end

                                            else
                                                objParts(idxPart,:) = [locPart.location(2:-1:1), 1];
                                            end
                                        else
                                            if(~isempty(locPart.location)) % Some empty annotations still...
                                                if(locPart.location(1) >= img_cols+1 || locPart.location(2) >= img_rows+1 || ...
                                                        locPart.location(1) < 0 || locPart.location(2) < 0)
                                                    objParts(idxPart,:) = [0, 0, 0];
                                                else
                                                    objParts(idxPart,:) = [locPart.location(2:-1:1), 1];
                                                end
                                            else
                                                objParts(idxPart,:) = [0, 0, 0];
                                            end
                                        end
                                    else
                                        objParts(idxPart,:) = [0, 0, 0];
                                    end
                                else
                                    objParts(idxPart,:) = [0, 0, 0];
                                end
                            end
                        end
                        if(sum(objParts(:,3)) == 0 && length(dataset.classes) == 100) % No single keypoint, remove
                            continue;
                        end
                        
%                         if(p == 2)
%                             [~, name, ~] = fileparts(listAnnotations{i});
%                             img = imread(['Z:\PhD\Data\Real\Multi\ObjectNet3D\Images\' name '.JPEG']);
%                             imshow(img);
%                             hold on;
%                             for idxP = 1:size(objParts,1)
%                                 parts = objParts;
%                                 if(parts(idxP,3) > 0)
%                                     scatter(parts(idxP,2), parts(idxP,1), 40, 'MarkerFaceColor', [0.9 0.9 0.2], 'MarkerEdgeColor',[0.9 0.9 0.0], 'LineWidth', 0.1);
%                                     if((parts(idxP,1) ~= 0 && parts(idxP,2) ~= 0))
%                                         text(parts(idxP,2)+1, parts(idxP,1)+1, nameParts{idxP}, 'HorizontalAlignment', 'left', 'Color', [0.9 0.9 0.2], 'FontSize', 10);
%                                     end
%                                 end
%                             end
%                             i
%                             hold off;
%                             keyboard;
%                         end
                        
%                         hold off;
%                         fprintf('id: %d\n', i);
%                         keyboard;
                        if(~strcmpi(nameDataset,'objectnet')) % Pascal3d and ImageNet3D
                            if(0) % p == 2) % Tulsiani et al.
                                if(~isfield(objAnno.viewpoint,'azimuth'))
                                    azimuth = [azimuth; objAnno.viewpoint.azimuth_coarse];
                                else
                                    azimuth = [azimuth; objAnno.viewpoint.azimuth];
                                end

                                if(~isfield(objAnno.viewpoint,'elevation'))
                                    elevation = [elevation; objAnno.viewpoint.elevation_coarse];
                                else
                                    elevation = [elevation; objAnno.viewpoint.elevation];
                                end

                                if(~isfield(objAnno.viewpoint,'theta'))
                                    plane = [plane; 0];
                                else
                                    plane = [plane; objAnno.viewpoint.theta];
                                end
                            else % if(p == 1)
                                if(objAnno.viewpoint.distance == 0)
                                    azimuth = [azimuth; objAnno.viewpoint.azimuth_coarse];
                                else
                                    azimuth = [azimuth; objAnno.viewpoint.azimuth];
                                end

                                if(objAnno.viewpoint.distance == 0)
                                    elevation = [elevation; objAnno.viewpoint.elevation_coarse];
                                else
                                    elevation = [elevation; objAnno.viewpoint.elevation];
                                end

                                if(objAnno.viewpoint.distance == 0)
                                    plane = [plane; 0];
                                else
                                    plane = [plane; objAnno.viewpoint.theta];
                                end
                            end
                        else % ObjectNet3D
                            if(objAnno.viewpoint.distance == 0)
                                az = objAnno.viewpoint.azimuth_coarse;
                                if(az < 0)
                                    azimuth = [azimuth; az + 360];
                                else
                                    azimuth = [azimuth; az];
                                end
                            else
                                if(~isfield(objAnno.viewpoint,'azimuth') || isempty(objAnno.viewpoint.azimuth))
                                    az = objAnno.viewpoint.azimuth_coarse;
                                else
                                    az = objAnno.viewpoint.azimuth;
                                end
                                if(az < 0)
                                    azimuth = [azimuth; az + 360];
                                else
                                    azimuth = [azimuth; az];
                                end
                            end
                            if(objAnno.viewpoint.distance == 0)
                                elevation = [elevation; objAnno.viewpoint.elevation_coarse];
                            else
                                if(~isfield(objAnno.viewpoint,'elevation') || isempty(objAnno.viewpoint.elevation))
                                    elevation = [elevation; objAnno.viewpoint.elevation_coarse];
                                else
                                    elevation = [elevation; objAnno.viewpoint.elevation];
                                end
                            end
                            plane = [plane; objAnno.viewpoint.theta];
                        end
%                         if(p == 2)
%                             if(plane(end) == 0 && elevation(end) == 0 && azimuth(end) == 0)
%                                 % imshow(imread([input.PATH_DATA 'Real/Multi/PASCAL_VOC12/train/JPEGImages/' imgAnnotation.record.filename]));
%                                 % keyboard;
%                                 continue;
%                             end
%                         end
                        parts = [parts; objParts];
                        classes = [classes; objAnno.class];
                        bb = objAnno.bbox;
                        annotations = [annotations; bb(2), bb(1), bb(4)-bb(2), bb(3)-bb(1)];
                        distance = [distance; objAnno.viewpoint.distance];
                        camera.focal = [camera.focal; objAnno.viewpoint.focal];
                        camera.px = [camera.px; objAnno.viewpoint.px];
                        camera.py = [camera.py; objAnno.viewpoint.py];
                        camera.viewport = [camera.viewport; objAnno.viewpoint.viewport];
                        idx = idx + 1;                    
                        
                        if p == 1
                            total_train(ismember(dataset.classes,objAnno.class)) = ...
                                total_train(ismember(dataset.classes,objAnno.class)) + 1;
                        else
                            total_test(ismember(dataset.classes,objAnno.class)) = ...
                                total_test(ismember(dataset.classes,objAnno.class)) + 1;
                        end
                        
                    end
                    % Check if already contained and get idx if so (add imgId)
                    if(idx > 0)
                        ids = [ids; idxPos(i)*ones(idx,1)];
                    end
                    if(strcmpi(nameDataset,'objectnet'))
                        if(mod(i,2500) == 0)
                            fprintf('annotations num. %d/%d\n',i,length(listAnnotations));
                        end
                    end
                end

                % Remove bad annotations (hard coded)
                if(strcmpi(nameDataset,'imagenet'))
                    if(strcmpi(class,'motorbike') && strcmpi(phases{p},'train'))
                        if(dataset.isOcclusions)
                            annotations(197,:) = [10, 8, 279, 295];
%                         else
%                             annotations(14,:) = annotations(14,[3,4,1,2]);
                        end
                    end
                    if(strcmpi(class,'motorbike') && strcmpi(phases{p},'train'))
                        if(~dataset.isOcclusions)
                            annotations(114,:) = [10, 8, 279, 295];
                        end
                    end
                    if(strcmpi(class,'aeroplane') && strcmpi(phases{p},'train') && dataset.isOcclusions)
                        annotations(6,:) = [25 1 270 499];
                    end
                end

                % For ObjectNet3D: Take only selected classes
                if(strcmpi(nameDataset,'objectnet'))
                    for c_save = 1:length(dataset.classes)
                         path_mat_data = [PATH 'mat_data\' dataset.classes{c_save} '_' phases{p} strOcc '.mat'];
                        if(~exist(path_mat_data,'file'))
                            takeSamples = ismember(classes, dataset.classes{c_save});
                            auxData.annotations.imgId = ids(takeSamples);
                            auxData.annotations.classes = classes(takeSamples);
                            auxData.annotations.parts = parts(takeSamples,:);
                            auxData.annotations.BB = annotations(takeSamples,:);
                            auxData.annotations.vp.azimuth = azimuth(takeSamples);
                            auxData.annotations.vp.elevation = elevation(takeSamples);
                            auxData.annotations.vp.distance = distance(takeSamples);
                            auxData.annotations.vp.plane = plane(takeSamples);
                            auxData.annotations.camera.focal = camera.focal(takeSamples);
                            auxData.annotations.camera.px = camera.px(takeSamples);
                            auxData.annotations.camera.py = camera.py(takeSamples);
                            auxData.annotations.camera.viewport = camera.viewport(takeSamples);
                            auxData.partLabels = allParts(c_save);
                            if(strcmpi(dataset.classes{c_save},'eyeglasses'))
                                aux_labels = auxData.partLabels{1};
                                aux_labels{15} = 'endpiece_right2';
                                auxData.partLabels{1} = aux_labels;
                            end
                            createDir([PATH 'mat_data\']);
                            save(path_mat_data, 'auxData', '-v7.3');
                        end
                    end
                    isFirstPass = false;
                else
                    auxData.annotations.imgId = ids;
                    auxData.annotations.classes = classes;
                    auxData.annotations.parts = parts;
                    auxData.annotations.BB = annotations;
                    auxData.annotations.vp.azimuth = azimuth;
                    auxData.annotations.vp.elevation = elevation;
                    auxData.annotations.vp.distance = distance;
                    auxData.annotations.vp.plane = plane;
                    auxData.annotations.camera.focal = camera.focal;
                    auxData.annotations.camera.px = camera.px;
                    auxData.annotations.camera.py = camera.py;
                    auxData.annotations.camera.viewport = camera.viewport;
                    auxData.partLabels = allParts;
                    createDir([PATH 'mat_data\']);
                    save(path_mat_data, 'auxData', '-v7.3');
                end
            end

            if(strcmpi(phases{p},'train'))
                data = appendData(nameDataset, data, auxData);
            else % test
                testData = appendData(nameDataset, testData, auxData);
            end
        
        end        

    end
    
%     input.targetDataset.isOcclusions = aux_occ;
    
    % Fix BBs out of bounds
	data.annotations.BB(data.annotations.BB < 1) = 1;
    testData.annotations.BB(testData.annotations.BB < 1) = 1;
    
%     for i = 1:length(dataset.classes)
%         fprintf('[train] %s: %d rejected, %d saved\n', dataset.classes{i}, numRejectedKps_train(i), total_train(i));
%         fprintf('[test] %s: %d rejected, %d saved\n', dataset.classes{i}, numRejectedKps_test(i), total_test(i));
%     end
%     keyboard;

    % Info files: ( size BBs, vp % kps instances)
%     isPlot = false;
%     gaussian = fspecial('gaussian', [1 5]);
%     phase = {'TRAIN', 'TEST'};
%     for i = 1:length(dataset.classes)
%         class = dataset.classes{i};
%         for p = 1:length(phase)
%             if(p == 1) % train
%                 pData = data;
%             else % test
%                 pData = testData;
%             end
%             idxs = ismember(pData.annotations.classes, class);
%             % -> BB info
%             bb = pData.annotations.BB(idxs,:);
%             num = size(bb,1);
%             row = [mean(bb(:,3)) std(bb(:,3))];
%             col = [mean(bb(:,4)) std(bb(:,4))];
%             fprintf('[%s]\n%s %d samples, row: %.1f,%.1f col: %.1f,%.1f\n', class, phase{p}, num, row, col);
%             % -> VP info
% %             az = round(pData.annotations.vp.azimuth(idxs,:));
% %             az(az > 360-8) = az(az > 360-8) - 360;
% %             [elems_az, centres_az] = hist(az,0:15:359);            
% %             if(isPlot)
% %                 figure; bar(centres_az,elems_az); axis([0 360 0 inf]); title([class ' Azimuth:'])
% %             end
% %             fprintf('- azimuth:'); fprintf(' %g', elems_az);
% %             elems_az = conv([elems_az(end-1:end), elems_az, elems_az(1:2)], gaussian, 'valid');
% %             centres_az(elems_az < length(az)*0.01) = [];
% %             fprintf('\n[ '); fprintf('%g ', centres_az); fprintf(']\n');
% %             el = round(pData.annotations.vp.elevation(idxs,:));
% %             [elems_el, centres_el] = hist(el,-90:15:89);
% %             if(isPlot)
% %                 figure; bar(centres_el,elems_el); axis([-90 90 0 inf]); title([class ' Elevation:'])
% %             end
% %             fprintf('- elevation:'); fprintf(' %g', elems_el);
% %             elems_el = conv([elems_el(1:2), elems_el, elems_el(end-1:end)], gaussian, 'valid');
% %             centres_el(elems_el < length(el)*0.01) = [];
% %             fprintf('\n[ '); fprintf('%g ', centres_el); fprintf(']\n');
% %             th = round(pData.annotations.vp.plane(idxs,:));
% %             th(th > 180-8) = th(th > 180-8) - 360;
% %             [elems_th, centres_th] = hist(th,-180:15:179);
% %             if(isPlot)
% %                 figure; bar(centres_th,elems_th); axis([-180 180 0 inf]); title([class ' InPlan:'])
% %             end
% %             fprintf('- inPlane:'); fprintf(' %g', elems_th);
% %             elems_th = conv([elems_th(end-1:end), elems_th, elems_th(1:2)], gaussian, 'valid');
% %             centres_th(elems_th < length(th)*0.01) = [];
% %             fprintf('\n[ '); fprintf('%g ', centres_th); fprintf(']\n');
%             % -> KPs info
% %             parts = pData.partLabels{i};
% %             infoParts = pData.annotations.parts(idxs);
% %             kp_prob = squeeze(sum(reshape(cell2mat(infoParts), [length(infoParts) length(parts) 3])));
% %             kp_prob = kp_prob(:,3) / length(infoParts);
% %             for idpart = 1:length(parts)
% %                 fprintf('%s %.2f ', parts{idpart}, kp_prob(idpart));
% %             end
% %             fprintf('\n');
%         end
%     end
%     keyboard;

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

function data = appendData(dataset, data, auxData)

    if(~strcmpi(dataset,'imagenet'))
        data.annotations.imgId = [data.annotations.imgId; auxData.annotations.imgId];
        data.imgPaths = sort_nat(unique([data.imgPaths; auxData.imgPaths]));
    else
        newImgIds = length(data.imgPaths)+auxData.annotations.imgId; % -min(auxData.annotations.imgId)+1;
        data.annotations.imgId = [data.annotations.imgId; newImgIds];
        data.imgPaths = [data.imgPaths; auxData.imgPaths];
    end    
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
%     if(strcmpi(dataset,'imagenet'))
%         keyboard;
%     end
end

function isBad = isNonConsistentSample(class, nameParts)
    isBad = (strcmpi(class, 'aeroplane') && length(nameParts) ~= 8) || ...
        (strcmpi(class, 'backpack') && length(nameParts) > 13) || ...
        (strcmpi(class, 'bicycle') && length(nameParts) ~= 11) || ...
        (strcmpi(class, 'boat') && length(nameParts) ~= 7) || ...
        (strcmpi(class, 'bottle') && length(nameParts) ~= 7) || ...
        (strcmpi(class, 'bucket') && length(nameParts) ~= 8) || ...
        (strcmpi(class, 'bed') && length(nameParts) ~= 20) || ...
        (strcmpi(class, 'bus') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'cabinet') && length(nameParts) ~= 16) || ...
        (strcmpi(class, 'cap') && length(nameParts) ~= 13) || ...
        (strcmpi(class, 'car') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'cellphone') && length(nameParts) ~= 16) || ...
        (strcmpi(class, 'chair') && length(nameParts) ~= 10) || ...
        (strcmpi(class, 'cup') && length(nameParts) ~= 17) || ...
        (strcmpi(class, 'diningtable') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'door') && length(nameParts) ~= 6) || ...
        (strcmpi(class, 'guitar') && length(nameParts) ~= 18) || ...
        (strcmpi(class, 'kettle') && length(nameParts) ~= 16) || ...
        (strcmpi(class, 'laptop') && length(nameParts) ~= 8) || ...
        (strcmpi(class, 'lighter') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'motorbike') && length(nameParts) ~= 10) || ...
        (strcmpi(class, 'pan') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'piano') && length(nameParts) ~= 21) || ...
        (strcmpi(class, 'plate') && length(nameParts) ~= 8) || ...
        (strcmpi(class, 'pot') && length(nameParts) ~= 13) || ...
        (strcmpi(class, 'remote_control') && length(nameParts) ~= 8) || ...
        (strcmpi(class, 'shoe') && length(nameParts) ~= 15) || ...
        (strcmpi(class, 'skateboard') && length(nameParts) ~= 10) || ...
        (strcmpi(class, 'sofa') && length(nameParts) ~= 10) || ...
        (strcmpi(class, 'suitcase') && length(nameParts) ~= 14) || ...
        (strcmpi(class, 'teapot') && length(nameParts) ~= 16) || ...
        (strcmpi(class, 'telephone') && length(nameParts) ~= 18) || ...
        (strcmpi(class, 'train') && length(nameParts) ~= 17) || ...
        (strcmpi(class, 'trash_bin') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'tvmonitor') && length(nameParts) ~= 8) || ...
        (strcmpi(class, 'washing_machine') && length(nameParts) ~= 12) || ...
        (strcmpi(class, 'watch') && length(nameParts) ~= 13);        
end

function isBad = isNonConsistentLabel(class, nameParts)

    if(isempty(nameParts))
        isBad = false;
    else
        isBad = (strcmpi(class, 'laptop') && strcmpi(nameParts{1}, 'front_bottom_left'));
    end

end

            % CHECK ERRORS in viewpoints
%             f = figure;
%             pivot = 90;
%             range = 30;
%             set(f, 'visible', 'off');
%             for i = 1:length(auxData.annotations.imgId)
%                 angle = auxData.annotations.viewpoints(i);
%                 if(angle > pivot+range || angle < pivot-range)
%                     continue;
%                 end
%                 img = imread(auxData.imgPaths{auxData.annotations.imgId(i)});
%                 BB = num2cell(auxData.annotations.BB(i,:));
%                 [row, col, height, width] = BB{:};
% %                 imshow(img);
%                 patch = cropBB(img, row, col, double(height), double(width), [128 256], 1);
% %                 patch = img(row:row+height-1,col:col+width-1,:);
%                 imshow(patch);
%                 hold on;
%                 text(10, 10, sprintf('%.1f',angle),'Color','w');
%                 sizeArrow = min([height width]/2);
%                 quiver(size(patch,2)/2,size(patch,1)/2,sin(angle*pi/180-pi)*sizeArrow,cos(angle*pi/180)*sizeArrow,...
%                     'Color','y','LineWidth',2,'MaxHeadSize',3);
% %                 rectangle('position', [col row width height], 'LineWidth', 1, 'EdgeColor', [0 0 1]);
% %                 sizeArrow = min([height width]/2);
% %                 quiver(col+width/2,row+height/2,sin(angle*pi/180-pi)*sizeArrow,cos(angle*pi/180)*sizeArrow,...
% %                     'Color','y','LineWidth',2,'MaxHeadSize',3);
% %                 text(col, row, sprintf('%.1f',angle),'Color',[1 1 1]);
%                 hold off;
%                 saveas(f, ['C:\Users\rmppanar\Desktop\imgs\img' num2str(i) '.png']);
%                 % keyboard;
%             end
%             keyboard;

            % Negatives....
%             if(c == length(dataset.classes))
%                 if(strcmpi(nameDataset,'pascal'))
%                     % - for Pascal:
%                     isNeg = ~ismember(allPaths,cumPos);
%                     negPaths = strcat(repmat({[input.PATH_DATA 'Real\Multi\PASCAL_VOC12\train\JPEGImages\']}, ...
%                         [sum(isNeg), 1]), allPaths(isNeg), repmat({'.jpg'}, [sum(isNeg), 1]));
%                 elseif(strcmpi(nameDataset,'imagenet'))
%                     % - for ImageNet:
%                     pathNeg = [input.PATH_DATA 'Real\Multi\ImageNet\ILSVRC2015\Data\DET\train\ILSVRC2013_train\'];
%                     folder = dir(pathNeg); classFolders = {folder.name};
%                     [~, ~, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
%                     folderPaths  = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1;
%                     folderPaths = sort_nat(classFolders(folderPaths)');
%                     % + random obj subfolder + random img
%                     idxRandom = 0;
%                     negPaths = {};
%                     while(idxRandom < 0)
%                         selObjPath = [folderPaths{randperm(length(folderPaths),1)} '\'];
%                         folder = dir([pathNeg selObjPath]); classFolders = {folder.name};
%                         [~, noms, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
%                         imgPath = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.JPEG'}), exts, 'UniformOutput', false)) == 1;
%                         imgPath = sort_nat(noms(imgPath)');
%                         randImg = randperm(length(imgPath),min(10,length(imgPath)));
%                         for negI = 1:length(randImg)
%                             if(~ismember(imgPath{randImg(negI)},cumPos))
%                                 idxRandom = idxRandom + 1;
%                                 negPaths = [negPaths; [pathNeg selObjPath imgPath{randImg(negI)} '.jpeg']];
%                             end
%                         end
%                     end
% 
%                 end
%                 auxData.imgPaths = [auxData.imgPaths; negPaths];
%             end

function printAnchor(name, locPart)

    locX = +Inf; locY = +Inf;
    if(~isempty(locPart.location))
        locX = locPart.location(1);
        locY = locPart.location(2);
    end

    status = +Inf;
    if(~isempty(locPart.status))
        status = locPart.status;
    end 
    
    fprintf('%s: status - %d [x %f y %f]\n', name, status, locX, locY);
    
end
