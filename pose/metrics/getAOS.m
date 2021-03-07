% Average Orientation Similarity
function cosSim = getAOS(gt, res, th, typeAngle)

    % typeAngle = 'rad', 'degree'
    if(~exist('typeAngle','var'))
        typeAngle = 'rad';
    end
    % conversion of angle metric if necessary
    if(strcmpi(typeAngle,'degree'))
       gt = gt/180.0*pi;
       res = res/180.0*pi;
    end
    
    rot_res = matRot(res(3),'plane')*matRot(-res(2),'elevation')*matRot(res(1),'azimuth');
    rot_gt = matRot(gt(3),'plane')*matRot(-gt(2),'elevation')*matRot(gt(1),'azimuth');
    delta = (1/sqrt(2))*norm(logm(rot_res'*rot_gt),'fro');
    cosSim = cos(delta);

end

