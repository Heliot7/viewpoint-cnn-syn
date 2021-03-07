function drawCube(origin, size, vp, f)

    if(~exist('f','var'))
        f = figure;
        set(f, 'Position', [2000 250 800 800]);
    else
        figure(f);        
    end
    hold on;
    
    % plot3([0,1],[0,0],[0,0],'Color','r','LineWidth',1.25);
    % plot3([0,0],[0,1],[0,0],'Color','g','LineWidth',1.25);
    % plot3([0,0],[0,0],[0,1],'Color','b','LineWidth',1.25);

    x = ([0 1 1 0 0 0; 1 1 0 0 1 1; 1 1 0 0 1 1; 0 1 1 0 0 0] - 0.5);
    y = ([0 0 1 1 0 0; 0 1 1 0 0 0; 0 1 1 0 1 1; 0 0 1 1 1 1] - 0.5);
    z = ([0 0 0 0 0 1; 0 0 0 0 0 1; 1 1 1 1 0 1; 1 1 1 1 0 1] - 0.5);

    R = matRot(-vp(3),'plane')*matRot(-vp(2),'elevation')*matRot(-vp(1),'azimuth');
    for idxFace = 1:6
        for idxVertex = 1:4
             newPoint = R*[x(idxVertex,idxFace); y(idxVertex,idxFace); z(idxVertex,idxFace)];
             x(idxVertex,idxFace) = newPoint(1);
             y(idxVertex,idxFace) = newPoint(2);
             z(idxVertex,idxFace) = newPoint(3);
        end
    end
    x = x / (max(x(:)) - min(x(:))) * size(1) + origin(1);
    y = y / (max(y(:)) - min(y(:))) * size(2) + origin(2);    
    % Plot only visible polygons
    for i = find(max(max(z)) == max(z)) % 1:6
        h = patch(x(:,i),y(:,i),z(:,i),'FaceColor', [0.75 0.1 0.1], 'FaceAlpha', 0.25, 'EdgeColor', 'b', 'LineWidth', 1.3);
        set(h,'edgecolor','k')
    end
    hold off;

end

function [y, z] = rotX(angle, y, z)

    angle = angle * 3.1416 / 180;
    y = y*cos(angle) - z*sin(angle);
    z = y*sin(angle) + z*cos(angle);

end

function [x, z] = rotY(angle, x, z)

    angle = angle * 3.1416 / 180;
    x = z*sin(angle) + x*cos(angle);
    z = z*cos(angle) - x*sin(angle);

end

function [x, y] = rotZ(angle, x, y)

    angle = angle * 3.1416 / 180;
    x = y*-sin(angle) + x*cos(angle);
    y = y*cos(angle) - x*sin(angle);

end
