function [input, data, features, testData, testFeatures] = getData(input, phase)

    data = []; testData = [];
    features = []; testFeatures = [];
    dataset = input.([phase 'Dataset']);
    switch(class(dataset))
        % Out Synthetic Data
        case 'Synthetic'
            data = getSyntheticData(input, dataset);
            input.sourceDataset.parts = data.partLabels;
        % ShapeNet Synthetic Data
        case 'ShapeNet'
            data = getShapeNet(input, dataset);
        % Multi-Object Detection
        case {'Pascal3D', 'ImageNet3D', 'ObjectNet3D'}
            if(strcmpi(class(dataset), 'Pascal3D') && dataset.addKps)
                [data, testData] = getPascalKps(input, dataset);
                input.targetDataset.parts = data.partLabels;
            else
                dataset3D = lower(class(dataset));
                [data, testData] = getObjectNet3D(input, dataset, dataset3D(1:end-2));
                input.targetDataset.parts = data.partLabels;
                if(strcmpi(class(dataset), 'Pascal3D') && dataset.addImageNet3D)
                    data_i3d = ImageNet3D;
                    data_i3d.isOcclusions = true; % tulsiani et al (false)
                    data_i3d.classes = input.targetDataset.classes;
                    [dataI3D, testDataI3D] = getObjectNet3D(input, data_i3d, 'imagenet');
                    data.annotations.imgId = [data.annotations.imgId; length(data.imgPaths)+dataI3D.annotations.imgId];
                    data.imgPaths = [data.imgPaths; dataI3D.imgPaths];
                    data.annotations.BB = [data.annotations.BB; dataI3D.annotations.BB];
                    data.annotations.classes = [data.annotations.classes; dataI3D.annotations.classes];
                    data.annotations.parts = [data.annotations.parts; dataI3D.annotations.parts];
                    data.annotations.vp.azimuth = [data.annotations.vp.azimuth; dataI3D.annotations.vp.azimuth];
                    data.annotations.vp.elevation = [data.annotations.vp.elevation; dataI3D.annotations.vp.elevation];
                    data.annotations.vp.distance = [data.annotations.vp.distance; dataI3D.annotations.vp.distance];
                    data.annotations.vp.plane = [data.annotations.vp.plane; dataI3D.annotations.vp.plane];
                    data.annotations.camera.px = [data.annotations.camera.px; dataI3D.annotations.camera.px];
                    data.annotations.camera.py = [data.annotations.camera.py; dataI3D.annotations.camera.py];
                    data.annotations.camera.focal = [data.annotations.camera.focal; dataI3D.annotations.camera.focal];
                    data.annotations.camera.viewport = [data.annotations.camera.viewport; dataI3D.annotations.camera.viewport];
                    data.annotations.imgId = [data.annotations.imgId; length(data.imgPaths)+testDataI3D.annotations.imgId];
                    data.imgPaths = [data.imgPaths; testDataI3D.imgPaths];
                    data.annotations.BB = [data.annotations.BB; testDataI3D.annotations.BB];
                    data.annotations.classes = [data.annotations.classes; testDataI3D.annotations.classes];
                    data.annotations.parts = [data.annotations.parts; testDataI3D.annotations.parts];
                    data.annotations.vp.azimuth = [data.annotations.vp.azimuth; testDataI3D.annotations.vp.azimuth];
                    data.annotations.vp.elevation = [data.annotations.vp.elevation; testDataI3D.annotations.vp.elevation];
                    data.annotations.vp.distance = [data.annotations.vp.distance; testDataI3D.annotations.vp.distance];
                    data.annotations.vp.plane = [data.annotations.vp.plane; testDataI3D.annotations.vp.plane];
                    data.annotations.camera.px = [data.annotations.camera.px; testDataI3D.annotations.camera.px];
                    data.annotations.camera.py = [data.annotations.camera.py; testDataI3D.annotations.camera.py];
                    data.annotations.camera.focal = [data.annotations.camera.focal; testDataI3D.annotations.camera.focal];
                    data.annotations.camera.viewport = [data.annotations.camera.viewport; testDataI3D.annotations.camera.viewport];
                end
            end
        % Human    
        case 'INRIA'
            [data, testData] = getINRIA(input, dataset);
        case 'CVC'
            data = getSyntheticCVC(input, dataset);
        case 'Daimler'
            [data, testData] = getDaimler(input, dataset);
        case 'Bahnhof'
            [data, testData] = getBahnhof(input, dataset);
        case 'TUD'
            [data, testData] = getTUD(input, dataset);
        case 'Caltech'
            [data, testData] = getCaltech(input, dataset);
        % Car
        case 'EPFL'
            [data, testData] = getEPFL(input, dataset);
        case 'KITTI'
            [data, testData] = getKITTI(input, dataset, phase);
        case 'NYC3DCARS'
            [data, testData] = getNYC3DCARS(input, dataset);
        case 'Freiburg'
            data = getFreiburg(input, dataset);
            testData = data;
        % Car + Bike
        case 'ObjCat3D'
            [data, testData] = getObjCat3D(input, dataset);
        % Multi-Object Classification
        case 'Saenko'
            [data, testData] = getSaenko(input, dataset, phase, 10);
%             [data, features, testData, testFeatures] = getSaenkoPrecomputed(input, dataset, phase, 10);
        case 'Office'
            [data, testData] = getSaenko(input, dataset, phase, 31);
%             [data, features, testData, testFeatures] = getSaenkoPrecomputed(input, dataset, phase, 31);
%             [data, features, testData, testFeatures] = getOfficePrecomputed_CNN(input, dataset, phase);
        case 'Testbed'
            [data, features, testData, testFeatures] = getTestbed(input, dataset, phase);
        case 'CrossDataset'
            % [data, features, testData, testFeatures] = getCrossDataset_raw(input, dataset, phase);
            [data, features, testData, testFeatures] = getCrossDataset(input, dataset, phase);
        case 'Sentiment'
            [data, features, testData, testFeatures] = getSentiment(input, dataset, phase);
        case 'MNIST'
            [data, features] = getMNIST(input, dataset);
        case 'SVHN'
            [data, features] = getSVHN(input, dataset);
        otherwise
            error('[[Caught ERROR: (phase %s) Wrong object class specified: %s]]', phase, class(dataset));
    end
    
    if(strcmpi(phase,'source') && input.isMirrored)
        newClass = cell(length(data.annotations.classes)*2,1);
        newClass(1:2:length(newClass)) = data.annotations.classes;
        newClass(2:2:length(newClass)) = data.annotations.classes;
        data.annotations.classes = newClass;
        newId = zeros(length(data.annotations.imgId)*2,1);
        newId(1:2:length(newId)) = data.annotations.imgId;
        newId(2:2:length(newId)) = data.annotations.imgId;
        data.annotations.imgId = newId;
    end
    
    % To comment (closed set -> open set)
    if(input.isRebuttal)
        if(~input.isOpenset && ~input.isWSVM)
            if(strcmpi(phase,'source'))
                input.sourceDataset.classes = input.sourceDataset.classes(1:10);
            elseif(strcmpi(phase,'target'))
                input.targetDataset.classes = input.sourceDataset.classes(1:10);
            end
        elseif(input.isOpenset)
            if(strcmpi(phase,'source'))
                input.sourceDataset.classes = [input.sourceDataset.classes(1:10), {'background'}];
            elseif(strcmpi(phase,'target'))
                input.targetDataset.classes = [input.sourceDataset.classes(1:10), {'background'}];
            end
        elseif(input.isWSVM)
            if(strcmpi(phase,'source'))
                input.sourceDataset.classes = input.sourceDataset.classes(1:10);
            elseif(strcmpi(phase,'target'))
                input.targetDataset.classes = [input.sourceDataset.classes(1:10), {'background'}];
            end
        end
    end
    
end

