function M = matRot(angle, plane)
    
    if(strcmpi(plane,'azimuth')) % Y
        M = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)];
    elseif(strcmpi(plane,'elevation')) % X
        M = [1 0 0; 0 cos(angle) -sin(angle); 0 sin(angle) cos(angle)];
    elseif(strcmpi(plane,'plane')) % Z 
        M = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1];
    end

end