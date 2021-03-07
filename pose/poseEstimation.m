function [res_vp, isCorrectVP, isCorrectVP_abs, isCorrect_pr, angleError, geocDist, PCK, PCK_cwm] = poseEstimation(input, data, netCaffe, sizePatch, classId, kpsId)

    % Get if AR is preserved or not (warping)
    keepAR = str2double(netCaffe.layers('data').get_layer_param('transform', 'keep_ar'));
    pad_pxl = str2double(netCaffe.layers('data').get_layer_param('transform', 'pad'));
    if(isempty(keepAR))
        keepAR = true;
    end

    % Visualisation plots 
    if(input.isShowPose)
        f_init = figure;
        set(f_init, 'visible', 'off');
        f_parts = figure;
        set(f_parts, 'visible', 'off');
        f_res = figure;
        set(f_res, 'visible', 'off');
        f_3D = figure;
        set(f_3D, 'visible', 'off');
    end
    
    patch_w = sizePatch(1);
    patch_h = sizePatch(2);
    
    % numImages trated as numInstances
    numInstances = min(input.numImages, length(data.annotations.imgId));
    results.annotations.imgId = data.annotations.imgId(1:numInstances);
    results.annotations.classes = data.annotations.classes(1:numInstances);
    results.annotations.BB = data.annotations.BB(1:numInstances,:);
    res_vp.azimuth = zeros(numInstances,1);
    res_vp.elevation = zeros(numInstances,1);
    res_vp.plane = zeros(numInstances,1);
    % [1..5] -> AZ4, AZ8, AZ16, AZ24, 3D-PI/6
    isCorrectVP = zeros(numInstances,5);
    isCorrectVP_abs = zeros(numInstances,5);
    angleError = zeros(numInstances,3);
    geocDist = zeros(numInstances,1);
    isCorrect_pr = cell(4,1);
    vp_num = [4,8,16,24]; 
    for i = 1:4
        isCorrect_pr{i} = zeros(vp_num(i),2);
    end
    
    if(isfield(data,'partLabels'))
        numParts = length(data.partLabels{1});
        initKps = sum(cellfun(@length, input.targetDataset.parts(1:classId)))+classId+1;
    else
        numParts = 10;
        initKps = 1;
    end
    % numInstances = 25;
    PCK = zeros(numParts,2); % col 2 = countPCK
    % (1) > 0.2 OK found (3) > 0.2 NO misplaced (4) < 0.2 NO missed 
    % (2) < 0.2 OK it is not there (5) > 0.2 should not be there
    PCK_cwm = zeros(numParts,6); % col 6 = countPCK
    PCK_scores = zeros(numInstances,numParts);
    idxImage = -1;
    for idx = 1:numInstances

        t = tic;
        fprintf('Instance num. %d/%d ... ', idx, numInstances);
        
        % Load new image
        if(results.annotations.imgId(idx) ~= idxImage)
            idxImage = results.annotations.imgId(idx);
            img = imread(data.imgPaths{idxImage});
            img = grey2rgb(img);
        end
        BB_orig = round(results.annotations.BB(idx,:));
        BB_aux = num2cell(BB_orig);
        [row_orig, col_orig, height_orig, width_orig] = BB_aux{:};
        % Check max number of stages
        nstage = 1;
        for idxStage = 1:6 % Stage = 6 is maximum
            string_to_search = sprintf('stage%d', idxStage);
            blob_id_C = strfind(netCaffe.blob_names, string_to_search);
            blob_id = not(cellfun('isempty', blob_id_C));
            if(sum(blob_id) > 0)
                nstage = idxStage;
            else
                break;
            end
        end
        % Scale selection
        scale_min = str2num(netCaffe.layers('data').get_layer_param('transform', 'scale_min'));
        scale_max = str2num(netCaffe.layers('data').get_layer_param('transform', 'scale_max'));
        scales = scale_min:0.1:scale_max;
        scores = cell(nstage, length(scales)); % 6 stages in CPM
        scores_patch = cell(nstage, length(scales));
        viewpoint = zeros(3,1);
        score_az = []; score_el = []; score_th = [];
        % scales = 1.0;
        for idxScale = 1:length(scales)
            
            % Adapt image and BB (w.r.t scale or warping)            
            % Resize to input CNN size (default: 368x368)
            if(keepAR)
                init_patch = uint8(128*ones(patch_h,patch_w,3));
                patch_mid = round([size(init_patch,1)/2, size(init_patch,2)/2]);
                % Phase 1
                crop_x = patch_w - pad_pxl*2;
                crop_y = patch_h - pad_pxl*2;
                ratio_w = crop_x / width_orig;
                ratio_h = crop_y / height_orig;
                ratio = min(ratio_w, ratio_h);
                scale = ratio*scales(idxScale);
                image_s = imresize(img, scale);
                scale_s = [size(image_s,1), size(image_s,2)];
                BB = num2cell(floor(BB_orig * scale));
                [row, col, height, width] = BB{:};
                cRow = row + 1 + ceil(height/2);
                r0_s = cRow - min(cRow-1, patch_mid(1)-1);
                r1_s = cRow + min(scale_s(1)-cRow, patch_mid(1));
                cCol = col + 1 + ceil(width/2);
                c0_s = cCol - min(cCol-1, patch_mid(2)-1);
                c1_s = cCol + min(scale_s(2)-cCol, patch_mid(2));
                % Phase 2
                r0 = patch_mid(1) - (cRow - r0_s);
                r1 = patch_mid(1) + (r1_s - cRow);
                c0 = patch_mid(2) - (cCol - c0_s);
                c1 = patch_mid(2) + (c1_s - cCol);
                init_patch(r0:r1,c0:c1,:) = image_s(r0_s:r1_s,c0_s:c1_s,:);
            else
                % Phase 1
                scale_warp = patch_w / ((patch_w - pad_pxl*2) * scales(idxScale));
                half_height = height_orig / 2;
                half_width = width_orig / 2;
                center = [col_orig + half_width, row_orig + half_height];
                bb_0_pre = [round(center(1) - half_width * scale_warp), round(center(2) - half_height * scale_warp)];
                bb_1_pre = [round(center(1) + half_width * scale_warp), round(center(2) + half_height * scale_warp)];
                unclipped_height = bb_1_pre(2) - bb_0_pre(2) + 1;
                unclipped_width = bb_1_pre(1) - bb_0_pre(1) + 1;
                pad_x1 = max(0, -bb_0_pre(1));
                pad_y1 = max(0, -bb_0_pre(2));
                bb_0 = [max(1, bb_0_pre(1)), max(1, bb_0_pre(2))];
                bb_1 = [min(size(img,2), bb_1_pre(1)), min(size(img,1), bb_1_pre(2))];
                clipped_height = bb_1(2) - bb_0(2) + 1;
                clipped_width = bb_1(1) - bb_0(1) + 1;
                scale_x = patch_w / unclipped_width;
                scale_y = patch_h / unclipped_height;
                crop_width = round(clipped_width*scale_x);
                crop_height = round(clipped_height*scale_y);
                pad_x1 = round(pad_x1*scale_x);
                pad_y1 = round(pad_y1*scale_y);
                pad_h = pad_y1;
                pad_w = pad_x1;
                if (pad_y1 + crop_height > patch_h)
                    crop_height = patch_h - pad_y1;
                end
                if (pad_x1 + crop_width > patch_w)
                    crop_width = patch_w - pad_x1;
                end
                % Phase 2
                init_patch = uint8(128*ones(patch_h,patch_w,3));
                tmp = imresize(img(bb_0(2):bb_1(2),bb_0(1):bb_1(1),:), [crop_height crop_width], 'bilinear', 'antialiasing', false);
                init_patch(pad_h+(1:crop_height), pad_w+(1:crop_width), :) = tmp; 
            end
%             figure(99); imshow(init_patch);
%             keyboard;
            
            % CNN pre-processing
            patch = (single(init_patch)-127.5)/255.0;
            patch = patch(:, : , [3, 2, 1]);
            patch = permute(patch, [2, 1, 3]);
            patch_mirrored = patch(end:-1:1,:,:);
            
            % Gaussian map for object localisation in input patch (normalised)
            if(nstage > 1)
                % sigma_center in training = 21
                gaussMap = fspecial('gaussian', [patch_h, patch_w], 21);
                gaussMap = single(gaussMap / max(max(gaussMap)));
                patch = cat(3, patch, gaussMap);
                patch_mirrored = cat(3, patch_mirrored, gaussMap);
                
                % Visualisation
%                 f_init = figure;
%                 visualise_input(f_init, image_s, init_patch, gaussMap, col, row, width, height);
            
                % Run CNN for Keypoint estimation
                part_bin = str2double(netCaffe.layers('data').get_layer_param('transform', 'num_parts')) + 1; % + bg
                redBlob = 0;
                try
                    netCaffe.forward({patch});
                catch
                    netCaffe.forward({patch; classId; kpsId});
                    redBlob = 1;
                end
                for idxStage = nstage % 1:nstage
                    string_to_search = sprintf('stage%d', idxStage);
                    blob_id_C = strfind(netCaffe.blob_names, string_to_search);
                    blob_id = find(not(cellfun('isempty', blob_id_C)));
                    if(~isempty(blob_id))
                        blob_id = blob_id(end)-redBlob;
                        scores{idxStage, idxScale} = permute(netCaffe.blob_vec(blob_id).get_data(), [2 1 3]);
                        % Get channels of corresponding class
                        score_class = scores{idxStage, idxScale};
%                         score_class = score_class(:,:,part_bin*classId+1:part_bin*classId+numParts+1); % + bg
                        score_class = score_class(:,:,initKps:initKps+numParts); % + bg(+1) included
                        scores{idxStage, idxScale} = score_class;
                    end
                    scores{idxStage,idxScale} = imresize(scores{idxStage,idxScale}, [patch_h patch_w]);
                end
                try
                    netCaffe.forward({patch_mirrored});
                catch
                    netCaffe.forward({patch_mirrored;classId; kpsId});
                end
                rel_parts = findKpsPerm(data.partLabels{1});
                for idxStage = nstage % 1:nstage
                    string_to_search = sprintf('stage%d', idxStage);
                    blob_id_C = strfind(netCaffe.blob_names, string_to_search);
                    blob_id = find(not(cellfun('isempty', blob_id_C)));
                    if(~isempty(blob_id))
                        blob_id = blob_id(end)-redBlob;
                        scores_mirrored = permute(netCaffe.blob_vec(blob_id).get_data(), [2 1 3]);
                        % Get channels of corresponding class
%                         scores_mirrored  = scores_mirrored(:,:,part_bin*classId+1:part_bin*classId+numParts+1); % + bg
                        scores_mirrored = scores_mirrored(:,:,initKps:initKps+numParts);
                        scores_mirrored  = scores_mirrored(:,end:-1:1,:);
                        scores_mirrored = imresize(scores_mirrored, [patch_h patch_w]);
                        % Find flip correspondences and merge with original score
                        for idxSwap = 1:max(rel_parts)
                            rels = find(rel_parts == idxSwap);
                            if(length(rels) > 1)
                                aux_score = scores_mirrored(:,:,rels(1));
                                scores_mirrored(:,:,rels(1)) = scores_mirrored(:,:,rels(2));
                                scores_mirrored(:,:,rels(2)) = aux_score;
                            end
                        end
                        scores{idxStage, idxScale} = (scores{idxStage, idxScale} + scores_mirrored) / 2;
                    end
                end 
                % Adapt heatmaps to original image size
                for idxStage = nstage % 1:nstage
                    score_post = scores{idxStage, idxScale};
                    if(keepAR)
                        scores_s = zeros(size(image_s,1), size(image_s,2), size(score_post,3));
                        scores_s(:,:,end) = 1;
                        score_post = score_post(r0:r1,c0:c1,:);
                        % Rescale cropped bb
                        cRow = cRow - r0_s + 1;
                        cCol = cCol - c0_s + 1;
                        extraP_x0 = max(0, 1 - (cRow - ceil(height/2)));
                        extraP_x1 = max(0, (cRow + floor(height/2) - 1) - size(score_post,1));
                        extraP_y0 = max(0, 1 - (cCol - ceil(width/2)));
                        extraP_y1 = max(0, (cCol + floor(width/2) - 1) - size(score_post,2));
                        aux_scores = score_post(cRow - ceil(height/2) + extraP_x0:cRow + floor(height/2) - 1 + extraP_x1, ...
                            cCol - ceil(width/2) + extraP_y0:cCol + floor(width/2) - 1 + extraP_y1,:);
                        score_post_patch = zeros(size(aux_scores) + [extraP_x0 + extraP_x1, extraP_y0 + extraP_y1, 0]);
                        score_post_patch(extraP_x0 + 1:end - extraP_x1, extraP_y0 + 1:end - extraP_y1,:) = aux_scores;
                        score_post_patch = imresize(score_post_patch, [height_orig, width_orig], 'bilinear', 'antialiasing', false);
                        scores_patch{idxStage,idxScale} = score_post_patch;
                        scores_s(r0_s:r1_s,c0_s:c1_s,:) = score_post;
                        scores_s = imresize(scores_s, [size(img,1) size(img,2)], 'bilinear', 'antialiasing', false);
                        scores{idxStage, idxScale} = scores_s;
                    else % warping
                        orig_scores = zeros(size(img,1), size(img,2), size(score_post,3));
                        orig_scores(:,:,end) = 1;
                        score_post = score_post(pad_h+(1:crop_height), pad_w+(1:crop_width), :); 
                        score_post_patch = imresize(score_post, [clipped_height clipped_width], 'bilinear', 'antialiasing', false);
                        diff_0 = bb_0_pre - bb_0;
                        score_bb_0 = [1 1] + diff_0;
                        diff_1 = bb_1_pre - bb_1;
                        score_bb_1 = [size(score_post_patch,2) size(score_post_patch,1)] + diff_1;
                        center = round((score_bb_1 + score_bb_0) / 2);
                        BB = [center(2:-1:1) - [half_height half_width] center(2:-1:1) + [half_height half_width]];
                        score_post_patch = score_post_patch(BB(1):BB(3)-1,BB(2):BB(4)-1,:);
                        scores_patch{idxStage,idxScale} = score_post_patch;
                        orig_scores(row_orig:row_orig+height_orig-1,col_orig:col_orig+width_orig-1,:) = score_post_patch;
                        scores{idxStage,idxScale} = orig_scores;
                    end
                end
%                 visualise_parts(f_parts, img, input.targetDataset.parts{1}, scores{nstage,idxScale});
%                 keyboard;
            end
            
            % Viewpoint estimation
            if(~strcmpi(input.cnnPoseTypeCNN, ''))
                nstage = 1;
                try
                    netCaffe.forward({patch});
                catch
                    netCaffe.forward({patch; classId; kpsId});
                end
                size_bin_vp = str2num(netCaffe.layers('data').get_layer_param('transform', 'size_bin_vp'));
                onlyAZ = str2num(netCaffe.layers('data').get_layer_param('transform', 'only_azimuth'));
                typeVP = 'sep'; % joint, sep, all
                if(sum(not(cellfun('isempty', strfind(netCaffe.blob_names, ['fc1_stage_vp' num2str(nstage)])))) > 0)
                    typeVP = 'joint';
                end
                if(strcmpi(typeVP,'joint'))
                    blob_id = not(cellfun('isempty', strfind(netCaffe.blob_names, ['fc1_stage_vp' num2str(nstage)])));
                    score_vp = netCaffe.blob_vec(blob_id).get_data();
                    if(strcmpi(input.cnnPoseTypeCNN,'reg'))
                        numBins = 6;
                        if(onlyAZ)
                            numBins = 2;
                        end
                        score_az = [score_az, score_vp(classId*numBins+1:classId*numBins+numBins)];
                    elseif(strcmpi(input.cnnPoseTypeCNN,'class'))
                        % Get az, el and th
                        az_bins = round(360/size_bin_vp);
                        if(onlyAZ)
                            dim_class = az_bins;
                        else
                            el_bins = round(180/size_bin_vp) + 1;
                            % el_bins = round(180/size_bin_vp);
                            th_bins = round(360/size_bin_vp);
                            dim_class = az_bins + el_bins + th_bins;
                        end
                        score_az = [score_az, score_vp(classId*dim_class+1:classId*dim_class+az_bins)];
                        if(~onlyAZ)
                            score_el = [score_el, score_vp(classId*dim_class+az_bins+1:classId*dim_class+az_bins+el_bins)];
                            score_th = [score_th, score_vp(classId*dim_class+az_bins+el_bins+1:classId*dim_class+dim_class)];
                        end
                    end
                % Separate should give the same results, not necessary
                elseif(strcmpi(typeVP,'sep')) % fc1_stage_vp_X1
                    if(strcmpi(input.cnnPoseTypeCNN,'reg'))
                    elseif(strcmpi(input.cnnPoseTypeCNN,'class'))
                        % -> Azimuth
                        blob_id = not(cellfun('isempty', strfind(netCaffe.blob_names, ['fc1_stage_vp_az' num2str(nstage)])));
                        az = netCaffe.blob_vec(blob_id).get_data();
                        score_az = [score_az; az(360*classId+1:360*classId+360)];
                        if(~onlyAZ)
                            % -> Elevation
                            blob_id = not(cellfun('isempty', strfind(netCaffe.blob_names, ['fc1_stage_vp_el' num2str(nstage)])));
                            el = netCaffe.blob_vec(blob_id).get_data();
                            score_el = [score_el; el(180*classId+1:180*classId+180)];
                            % -> InPlane Rotation
                            blob_id = not(cellfun('isempty', strfind(netCaffe.blob_names, ['fc1_stage_vp_th' num2str(nstage)])));
                            th = netCaffe.blob_vec(blob_id).get_data();
                            score_th = [score_th; th(360*classId+1:360*classId+360)];
                        end
                    end
                end               

                % Mirrored (Test)
                try
                    netCaffe.forward({patch_mirrored});
                catch
                    netCaffe.forward({patch_mirrored;classId; kpsId});
                end
                % netCaffe.forward({patch_mirrored;classId; kpsId});
                blob_id = not(cellfun('isempty', strfind(netCaffe.blob_names, ['fc1_stage_vp' num2str(nstage)])));
                score_vp = netCaffe.blob_vec(blob_id).get_data();
                if(strcmpi(input.cnnPoseTypeCNN,'reg'))
                    numBins = 6;
                    if(onlyAZ)
                        numBins = 2;
                    end
                    reg_mirrored =  score_vp(classId*numBins+1:classId*numBins+numBins);
                    reg_mirrored(2) = -reg_mirrored(2);
                    if(numBins == 6)
                        reg_mirrored(6) = -reg_mirrored(6);
                    end
                    score_az(:,end) = (score_az(:,end) + reg_mirrored)/2;
                elseif(strcmpi(input.cnnPoseTypeCNN,'class'))
                    % Get az, el and th
                    az_bins = round(360/size_bin_vp);
                    if(onlyAZ)
                        dim_class = az_bins;
                    else
                        el_bins = round(180/size_bin_vp) + 1;
                        % el_bins = round(180/size_bin_vp);
                        th_bins = round(360/size_bin_vp);
                        dim_class = az_bins + el_bins + th_bins;
                    end
                    az_mirrored = score_vp(classId*dim_class+1:classId*dim_class+az_bins);
                    az_mirrored = az_mirrored([1, end:-1:2]);
                    % MIRROR
                    score_az(:,end) = (score_az(:,end) + az_mirrored)/2;
                    if(~onlyAZ)
                        el_mirrored = score_vp(classId*dim_class+az_bins+1:classId*dim_class+az_bins+el_bins);
                        % MIRROR
                        score_el(:,end) = (score_el(:,end) + el_mirrored)/2;
                        th_mirrored = score_vp(classId*dim_class+az_bins+el_bins+1:classId*dim_class+dim_class);
                        if(mod(th_bins,2) == 0)
                            th_mirrored = th_mirrored([1, end:-1:2]); 
                        else
                            th_mirrored = th_mirrored(end:-1:1);
                        end
                        % MIRROR
                        score_th(:,end) = (score_th(:,end) + th_mirrored)/2;
                    end  
                end
            end
            nstage = 6;
            
        end
        % Merge all scores for all scales
        final_scores = cell(nstage,1);
        final_scores_patch = cell(nstage,1);
        for idxStage = nstage % 1:nstage
            final_scores{idxStage} = zeros(size(scores{nstage,1}));
            final_scores_patch{idxStage} = zeros(size(scores_patch{nstage,1}));
            for idxScale = 1:length(scales)
                final_scores{idxStage} = final_scores{idxStage} + scores{idxStage,idxScale};
                final_scores_patch{idxStage} = final_scores_patch{idxStage} + scores_patch{idxStage,idxScale};
            end
            final_scores{idxStage} = final_scores{idxStage} / size(scores,2);
            final_scores_patch{idxStage} = final_scores_patch{idxStage} / size(scores_patch,2);
        end         
        % XY positions of found parts (cropped instance and whole image)
        score_post = final_scores{nstage};
        score_post_patch = final_scores_patch{nstage};
        pred_parts_cropped = zeros(size(score_post_patch,3)-1,2);
        pred_parts_orig = zeros(size(score_post,3)-1,2);
        for j = 1:size(score_post,3)-1 % not background
            score_part_patch = score_post_patch(:,:,j);
            [value, pos] = max(score_part_patch(:));
            % if(value > 0.15)
                [pred_parts_cropped(j,1), pred_parts_cropped(j,2)] = ind2sub(size(score_part_patch), pos);
                score_part_orig = score_post(:,:,j);
                [~, pos_orig] = max(score_part_orig(:));
                [pred_parts_orig(j,1), pred_parts_orig(j,2)] = ind2sub(size(score_part_orig), pos_orig);
            % end
        end
        if(nstage > 1)
            % Compute PCK for keypoint localisation
            gtObjParts = data.annotations.parts{idx};
            resObjParts = pred_parts_orig;
            thPart = 0.1*max(height_orig, width_orig);
            for idxP = 1:size(gtObjParts,1)
                if(gtObjParts(idxP,3))
                    PCK(idxP,2) = PCK(idxP,2) + 1;
                    if(resObjParts(idxP,1) ~= 0)
                        dist = norm(resObjParts(idxP,:) - gtObjParts(idxP,1:2));
                        if(dist <= thPart)
                            PCK(idxP,1) = PCK(idxP,1) + 1;
                        end
                    end
                end
                PCK_scores(idx,idxP) = PCK(idxP,1) / PCK(idxP,2);
            end        
            % Compute new metric with OK, Wrong, OKno, wrongNo and Missed
            thVal = 0.15;
            PCK_cwm(:,6) = PCK_cwm(:,6) + 1;
            for idxP = 1:size(gtObjParts,1)
                val = score_post(resObjParts(idxP,1), resObjParts(idxP,2), idxP);                
                if(gtObjParts(idxP,3) > 0)
                    dist = norm(resObjParts(idxP,:) - gtObjParts(idxP,1:2));
                    if(dist <= thPart && val >= thVal)
                        PCK_cwm(idxP,1) = PCK_cwm(idxP,1) + 1;   % found
                    elseif(val >= thVal)
                        PCK_cwm(idxP,3) = PCK_cwm(idxP,3) + 1;   % misplaced
                    else
                        PCK_cwm(idxP,4) = PCK_cwm(idxP,4) + 1;   % missed
                    end
                else % occluded or truncated
                    if(val < thVal)
                        PCK_cwm(idxP,2) = PCK_cwm(idxP,2) + 1; % occ/trunc seen
                    else
                        PCK_cwm(idxP,5) = PCK_cwm(idxP,5) + 1; % should not be there
                    end
                end
            end        
        end
        
        % Estimate exact viewpoint based on method
        if(strcmpi(input.cnnPoseTypeCNN, 'reg'))
            % Merge viewpoint results
            score_az = mean(score_az, 2);
            viewpoint(1) = atan2(score_az(2), score_az(1)) * 180 / 3.1416;
            if(viewpoint(1) < 0)
                viewpoint(1) = viewpoint(1) + 360;
            end
            if(length(score_az) == 6) % also elevation and in-plane rotation
                viewpoint(2) = atan2(score_az(4), score_az(3)) * 180 / 3.1416;
                viewpoint(3) = atan2(score_az(6), score_az(5)) * 180 / 3.1416;
            end
        elseif(strcmpi(input.cnnPoseTypeCNN,'class'))
            % score_az = max(score_az,[], 2);
            score_az = mean(score_az, 2);
            if(input.goCubicUpscale)
                score_az_aux = [score_az(end-1:end); score_az; score_az(1:2)];
                expand_az = repmat(score_az_aux, [1 size_bin_vp])';
                expand_az = expand_az(:);
                cubicFilter = cubicFilter1D(size_bin_vp);
                % gauss = gaussFilter1D(size_bin_vp*3, 10.0);
                expand_az2 = conv(expand_az, cubicFilter, 'same');
                expand_az3 = expand_az2(2*size_bin_vp+1:end-2*size_bin_vp);
                [~, az_conv] = max(expand_az3);
                az = mod(az_conv - floor(size_bin_vp/2.0) + 360, 360);
            else
                [~, old_az] = max(score_az);
                old_az = (old_az-1)*size_bin_vp; % Convert to 360° value
                az = old_az;
            end
            if(~onlyAZ) % also elevation and in-plane rotation
                % -> elevation
                offset_el = ((90/size_bin_vp) - floor(90/size_bin_vp)) * size_bin_vp;
                % score_el = max(score_el, [], 2);
                score_el = mean(score_el, 2);
                if(input.goCubicUpscale)
                    expand_el = interp1(linspace(0,1, numel(score_el)), score_el, linspace(0,1,181), 'nearest');
                    cubicFilter = cubicFilter1D(size_bin_vp);
                    expand_el2 = conv(expand_el(1:end-1), cubicFilter, 'same');
                    [~, el_conv] = max(expand_el2);
                    el = el_conv;
                else
                    [~, old_el] = max(score_el);
                    old_el = (old_el-1)*size_bin_vp + offset_el;
                    el = old_el;
                end
                % -> tilt
                offset_th = ((180/size_bin_vp) - floor(180/size_bin_vp)) * size_bin_vp;
                % score_th = max(score_th, [], 2);
                score_th = mean(score_th, 2);
                if(input.goCubicUpscale)
                    score_th_aux = [score_th(end-1:end); score_th; score_th(1:2)];
                    expand_th = repmat(score_th_aux, [1 size_bin_vp])';
                    expand_th = expand_th(:);
                    cubicFilter = cubicFilter1D(size_bin_vp);
                    expand_th2 = conv(expand_th, cubicFilter, 'same');
                    expand_th3 = expand_th2(2*size_bin_vp+1:end-2*size_bin_vp);
                    [~, th_conv] = max(expand_th3);
                    th = mod(th_conv - floor(size_bin_vp/2.0) + 360, 360);
                else
                    [~, old_th] = max(score_th);
                    old_th = (old_th-1)*size_bin_vp + offset_th;
                    th = old_th;
                end
                viewpoint = [az, el-90, th-180];
            else
                viewpoint = [az, 0, 0];
            end
        else
            viewpoint = [0, 0, 0]; % only keypoint estimation
        end
        res_vp.azimuth(idx) = viewpoint(1);
        res_vp.elevation(idx) = viewpoint(2);
        res_vp.plane(idx) = viewpoint(3);
        
        % Compute viewpoint accuracy
        if(~strcmpi(input.cnnPoseTypeCNN,''))
            % -> 3D viewpoint (AVP - Pascal3D)
            if(isfield(data.annotations.vp,'elevation'))
                gt = [data.annotations.vp.azimuth(idx), data.annotations.vp.elevation(idx), data.annotations.vp.plane(idx)];
            else
                gt = [data.annotations.vp.azimuth(idx), 0.0, 0.0];
            end
            [isCorrectVP(idx,:), isCorrectVP_abs(idx,:), isCorrect_pr, angleError(idx,:), geocDist(idx)] = ...
                getAVP(gt, viewpoint, isCorrect_pr, 'degree');
            % cosSim = getAOS(gt, viewpoint(:,end), pi/6, 'degree');
        end
        
        % Visualisation
        if(input.isShowPose)
            % -> Pose estimation (whole img)
            % visualise_parts(f_parts, img, input.targetDataset.parts{1}, final_scores{nstage});
            % visualise_results(f_parts, img, pred_parts_orig);
            % -> Pose estimation (cropped bb)
            if(nstage > 1)
                % patch = img(row_orig:row_orig+height_orig-1, col_orig:col_orig+width_orig-1,:);
                visualise_parts(f_parts, img, input.targetDataset.parts{1}, final_scores{nstage});
                visualise_results(f_parts, img, pred_parts_orig);
            else
                figure(f_parts);
                set(f_parts, 'Position', [2000 200 1500 900]);
            end
            % -> 3D viewpoint
            if(~strcmpi(input.cnnPoseTypeCNN,''))
                cam.px = data.annotations.camera.px(idx); cam.py = data.annotations.camera.py(idx);
                cam.focal = data.annotations.camera.focal(idx); cam.vp = data.annotations.camera.viewport(idx);
                visualise_3D(f_parts, img, cam, ...
                    [row_orig col_orig height_orig width_orig], viewpoint, size(pred_parts_orig,1));
            end
            keyboard;
            pause(0.5);
        end
        
        fprintf('in %.2fs\n', toc(t));
        % keyboard;
        
    end

%     figure;
%     PCK_scores(isnan(PCK_scores)) = 0;
%     plot(PCK_scores);
%     legend(input.targetDataset.parts{1},'Location','southwest')
%     title(sprintf('Avg. %.2f',mean(PCK(:,1)./PCK(:,2))));
%     keyboard;
    
end

function visualise_input(f, img, patch, gaussMap, col, row, width, height)
    
    figure(f);
    set(f, 'Position', [100 2000 1500 900]);
    % set(f, 'Position', get(0, 'ScreenSize'));
    subplot(1,3,1);
    imshow(img);
    rectangle('position', [col row width height], 'LineWidth', 1, 'EdgeColor', [0 0 1]);
    axis off; axis image;
    subplot(1,3,2);
    imshow(patch);
    axis off; axis image;
    subplot(1,3,3);
    imagesc(gaussMap);
    axis off; axis image;
    
end

function visualise_parts(f, img, strParts, score_post)

    figure(f);
    set(f, 'Position', [2000 200 1500 900]);
    numXrow = ceil(((length(strParts)+1)/4));
    for i = 1:length(strParts)+1
        subplot(numXrow,4,i);
        % imToShow = mat2im(score_post(:,:,i), jet(100), [0 max(max(score_post(:,:,i)))])*0.5 + (single(img)/255)*0.5;
        imToShow = mat2im(score_post(:,:,i), jet(100), [0 1])*0.5 + (single(img)/255)*0.5;
        imagesc(imToShow);
        if(i <= length(strParts))
            probPart = max(max(score_post(:,:,i)));
            strProbPart = sprintf('%.3f', probPart);
            if(probPart > 0.15)
                title([strrep(strParts{i}, '_', '\_') ' ' strProbPart], 'Color','b');
            else
                title([strrep(strParts{i}, '_', '\_') ' ' strProbPart]);
            end
        else % background
            title('background');
        end
        axis off; axis image;
    end

end

function visualise_results(f, img, parts)

    sizeSubPlot = length(findobj(f,'type','axes'));
    figure(f);
    set(f, 'Position', [2000 200 1500 900]);
    if(sizeSubPlot > 1)
        numXrow = ceil(((size(parts,1)+1)/4));
        subplot(numXrow,4,size(parts,1)+2);
    end
    imshow(img);
    axis image;
    hold on;
    % Draw found parts
    parts(parts(:,1) == 0,:) = [];
    if(~isempty(parts))
        scatter(parts(:,2), parts(:,1),'filled');
        % Minimum spannin tree for visualisation
        if(size(parts,1) > 1)
            connections = ones(size(parts,1)) - eye(size(parts,1));
            weights = zeros(size(parts,1));
            for elem1 = 1:size(parts,1)
                for elem2 = 1:size(parts,1)
                    weights(elem1, elem2) = norm(parts(elem1,:) - parts(elem2,:));
                end
            end
            [~, ST] = kruskal(connections, weights);
            plot([parts(ST(:,1),2); parts(ST(:,2),2)],[parts(ST(:,1),1); parts(ST(:,2),1)]);
        end
    end
    hold off;

end 

function visualise_3D(f, img, cam, bb, vp, numParts)

    % Setup camera matrix
    intParam = [cam.focal*cam.vp 0 0; 0 cam.focal*cam.vp 0; 0 0 1];
    vp_rad = vp / 180.0 * pi;
    extParam = matRot(vp_rad(3),'plane')*matRot(-vp_rad(2),'elevation')*matRot(vp_rad(1),'azimuth');

    % Setup Plot
    sizeSubPlot = length(findobj(f,'type','axes'));
    % f = figure;
    set(f, 'Position', [2000 200 1500 900]);
    if(sizeSubPlot > 1)
        numXrow = ceil(((numParts+1)/4));
        subplot(numXrow,4,numParts+3);
    end
    imshow(img);
    axis image;
    hold on;
    % drawCube([bb(2) + bb(4)/2, bb(1) + bb(3)/2], [bb(4), bb(3)], vp_rad, f);
    % hold on;
    
    % Setup quiver 3D
    dir = intParam*extParam*[0; 0; -1];
    dir = dir/norm(dir);
    cen = [bb(2)+bb(4)/2, bb(1)+bb(3)/2 1.0];
    quiver(cen(1), cen(2), dir(1), -1*dir(2), double(min(bb(3:4))/2),'Color',[1 1 0],'LineWidth',2.0,'MaxHeadSize',0.5);
    hold off;

end

function gaussFilter = gaussFilter1D(size, sigma)

    x = linspace(-size / 2, size / 2, size);
    gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
    gaussFilter = gaussFilter / sum (gaussFilter); % normalize

end

function cubicFilter = cubicFilter1D(binSize)

    cubicFilter = zeros(4*binSize+1,1);
    idx = 1;
    for i = -2:1/binSize:2
        x = abs(i);
        if(x >= 1) % 1 <= x < 2
            cubicFilter(idx) = -0.5*(x*x*x) + 2.5*(x*x) - 4*x + 2;
        else % |x| < 1
            cubicFilter(idx) = 1.5*(x*x*x) - 2.5*(x*x) + 1;
        end
        idx = idx + 1;
    end

end
