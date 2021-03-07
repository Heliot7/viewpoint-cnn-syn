function [AP, AVP, gtDetections, matches, resDetections, allMatches, allDetections] = evalAVP(dataGT, gt, BB, vp_det)

    isVP = false;
    if(exist('vp_det','var'))
        isVP = true;
    end

    nd = size(BB,1);
    tp = zeros(nd,6);
    fp = zeros(nd,6);
    % assign detections to ground truth objects
    gtDetections = false(size(gt,1),1);
    matches = zeros(nd,1);
    resDetections = false(nd,1);
    allDetections = false(nd,1);
    allMatches = zeros(nd,1);
    for d=1:nd

        % find ground truth image % remove full path
        % i=VOChash_lookup(hash,ids{d});
        i = BB(d,5);
        if isempty(i)
            error('unrecognized image "%s"', gt(d,5));
        elseif length(i)>1
            error('multiple image "%s"', gt(d,5));
        end

        % assign detection to ground truth object if any
        bb = BB(d,1:4);
        ovmax = -inf;
        idxAnno = find(gt(:,5) == BB(d,5));
        for j=1:length(idxAnno)% size(gt(i).BB,2)
            % bbgt=gt(i).BB(:,j);
            bbgt = gt(idxAnno(j),1:4);
            bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 & ih>0                
                % compute overlap as area of intersection / area of union
                ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                   (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                   iw*ih;
                ov=iw*ih/ua;
                if ov>ovmax
                    ovmax=ov;
                    jmax=j;
                end
            end
        end
        
        % assign detection as true positive/don't care/false positive
        notDetAnno_img = gtDetections(idxAnno);
        if ovmax >= 0.5
            % showDetection(dataGT.imgPaths{dataGT.annotations.imgId(idxAnno)}, bbgt, bb);
            if ~notDetAnno_img(jmax)
                % true positive
                tp(d,1) = 1;
                notDetAnno_img(jmax) = true;
                resDetections(d) = true;
                matches(d) = idxAnno(jmax);
                if(isVP)
                    for iVP = 1:5
                        if(vp_det(d,iVP))
                            tp(d,1+iVP) = 1;
                        else
                            fp(d,1+iVP) = 1;
                        end
                    end
                end
            else
                % false positive (multiple detection) for all AP and AVP
                fp(d,:) = [1,1,1,1,1,1];
            end
            % Include multiple detections
            allDetections(d) = true;
            allMatches(d) = idxAnno(jmax);
        else
            % false positive for all AP and AVP
            fp(d,:) = [1,1,1,1,1,1];
        end
        gtDetections(idxAnno,:) = notDetAnno_img;
    end
    
    % compute precision/recall
    fp_i = cumsum(fp(:,1));
    tp_i = cumsum(tp(:,1));
    rec = tp_i/size(gt,1); %npos;
    prec = tp_i./(fp_i+tp_i);
    AP = VOCap(rec,prec);
    % plot precision/recall
%     plot(rec,prec,'-');
%     grid;
%     xlabel 'recall'
%     ylabel 'precision'
%     title(sprintf('AP = %.3f', AP));    

    AVP = zeros(1,5);
    if(isVP)
        for i = 1:5
            fp_i = cumsum(fp(:,1+i));
            tp_i = cumsum(tp(:,1+i));
            % rec = tp_i/size(gt,1);
            prec = tp_i./(fp_i+tp_i);
            % Use recall from AP (keep the same, only changes precision)
            AVP(i) = VOCap(rec,prec);
        end
    end
    
end

function showDetection(path, bbgt, bb)

    imshow(imread(path));
    rectangle('position', [bbgt(1:2) bbgt(3:4)-bbgt(1:2)], 'LineWidth', 1, 'EdgeColor', [0 0 1]);
    rectangle('position',[bb(1:2) bb(3:4)-bb(1:2)], 'LineWidth', 1, 'EdgeColor', [0 1 0]);

end