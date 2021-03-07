function [isCorrect, isCorrectAbs, isCorrect_pr, angleError, delta]= getAVP(gt, res, isCorrect_pr, typeAngle)

    isCorrect = zeros(1,5);
    isCorrectAbs = zeros(1,5);
    angleError = zeros(1,3);
    
    % typeAngle = 'rad', 'degree'
    if(~exist('typeAngle','var'))
        typeAngle = 'rad';
    end
    
    if(strcmpi(typeAngle,'rad'))
       gt = gt*180/pi;
       res = res*180/pi;
    end
    isCorrect(1) = getBin(4, gt(1)) == getBin(4, res(1)); % 90° bins
    aux = isCorrect_pr{1};
    aux(getBin(4, gt(1)),isCorrect(1)+1) = aux(getBin(4, gt(1)),isCorrect(1)+1) + 1;
    isCorrect_pr{1} = aux;
    isCorrect(2) = getBin(8, gt(1)) == getBin(8, res(1)); % 45° bins
    aux = isCorrect_pr{2};
    aux(getBin(8, gt(1)),isCorrect(2)+1) = aux(getBin(8, gt(1)),isCorrect(2)+1) + 1;
    isCorrect_pr{2} = aux;
    isCorrect(3) = getBin(16, gt(1)) == getBin(16, res(1)); % 22.5° bins
    aux = isCorrect_pr{3};
    aux(getBin(16, gt(1)),isCorrect(3)+1) = aux(getBin(16, gt(1)),isCorrect(3)+1) + 1;
    isCorrect_pr{3} = aux;
    isCorrect(4) = getBin(24, gt(1)) == getBin(24, res(1)); % 15° bins
    aux = isCorrect_pr{4};
    aux(getBin(24, gt(1)),isCorrect(4)+1) = aux(getBin(24, gt(1)),isCorrect(4)+1) + 1;
    isCorrect_pr{4} = aux;
    deltaAz = min([abs(gt(1) - res(1)), abs(gt(1)+360 - res(1)), abs(gt(1)-360 - res(1))]);
    isCorrectAbs(1) = deltaAz < 90/2; % pi/4; % 90° bins
    isCorrectAbs(2) = deltaAz < 45/2; % pi/8; % 45° bins
    isCorrectAbs(3) = deltaAz < 22.5/2; % pi/16; % 22.5° bins
    isCorrectAbs(4) = deltaAz < 15/2; % pi/24; % 15° bins
    angleError(1) = deltaAz;
    angleError(2) = abs(gt(2) - res(2));
    angleError(3) = abs(gt(3) - res(3));
    
    % Conversion of angle metric if necessary
    if(strcmpi(typeAngle,'degree'))
       gt = gt/180.0*pi;
       res = res/180.0*pi;
    end
    
    %  Results
    rot_res = matRot(res(3),'plane')*matRot(-res(2),'elevation')*matRot(res(1),'azimuth');
    rot_res = angle2dcm_pau(res(1), res(2), res(3));
    rot_res2 = angle2dcm_pau(res(3), res(2)-pi/2, -res(1), 'ZXZ');
    % GT
    rot_gt = matRot(gt(3),'plane')*matRot(-gt(2),'elevation')*matRot(gt(1),'azimuth');
    rot_gt = angle2dcm_pau(gt(1), gt(2), gt(3));
    rot_gt2 = angle2dcm_pau(gt(3), gt(2)-pi/2, -gt(1), 'ZXZ');
    
%     R = rot_res'*rot_gt;
%     delta = (1/sqrt(2))*norm(logm(R),'fro');
    R2 = rot_res2'*rot_gt2;
    delta = (1/sqrt(2))*norm(logm(R2),'fro');

    isCorrect(5) = (delta < pi/6);
    isCorrectAbs(5) = (delta < pi/6);
    delta = delta*180.0/pi;
end

function idView = getBin(K, azimuth)

    idView = getAzimuthId(getDiscreteAzimuths(K, false), azimuth);

end