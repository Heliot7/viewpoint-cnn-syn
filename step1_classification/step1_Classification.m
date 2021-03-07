function [out_transfer, srcFeatures, out_classifiers] = step1_Classification(input, srcData, srcFeatures, tgtData, tgtFeatures, testData, testFeatures)

    if(input.dimPCA > 0.0 && (strcmpi(input.cnnModel,'AlexNet') || strcmpi(input.cnnModel,'VGG-16')))
        if(input.dimPCA > 1)
            if(strcmpi(input.cnnModel,'AlexNet'))
                minDims = 0.33;
            elseif(strcmpi(input.cnnModel,'VGG-16'))
                minDims = 0.25;
            end
        else
            minDims = round(size(srcFeatures,2)*input.dimPCA);
        end
        fprintf('PCA reduction to %d dims\n', minDims);
        tic;
        if(size(srcFeatures,1) < 4000)
            coeff = mPCA([srcFeatures; tgtFeatures]);
        else
            coeff = kPCA([srcFeatures; tgtFeatures], minDims);
        end
        toc;
%         keyboard;
        srcFeatures = srcFeatures*coeff(:,1:minDims);
        tgtFeatures = tgtFeatures*coeff(:,1:minDims);
        if(~isempty(testFeatures))
            testFeatures = testFeatures*coeff(:,1:minDims);
        end
    end
    
    out_classifiers = [];
    % Define labels involved in domain adaptation
    [srcIds, srcClasses, metadata] = getIdLabels(input.sourceDataset, srcData.annotations);
    [tgtIds, tgtClasses] = getIdLabels(input.targetDataset, tgtData.annotations, metadata);
    if(isfield(testData,'annotations'))
        testIds = getIdLabels(input.targetDataset, testData.annotations);
    else
        testIds = [];
    end

    % Compute Domain Adaptation
    transferLabels = [];
    if(input.isDA && (~strcmpi(class(input.sourceDataset), class(input.targetDataset)) || isprop(input.sourceDataset,'source')))

        if(strcmpi(input.typeDA, 'MMDT'))
            P = mPCA([srcFeatures; tgtFeatures]);
            testFeatures = testFeatures*P(:,1:input.dimPCA);
        end
        
        % Call DA        
        [srcFeatures, tgtFeatures, W, out_classifiers, transferLabels, testFeatures] = DA(input, ...
            srcData, srcClasses, srcIds, srcFeatures, ...
            tgtData, tgtClasses, tgtIds, tgtFeatures, testData, testIds, testFeatures);
%             input.input.featureInfo.featureDim = size(srcFeatures,2);

        if(strcmpi(input.typeDA, 'MMDT') && ~isempty(testFeatures))
            testFeatures = testFeatures*W(1:input.dimPCA,1:input.dimPCA);
            testFeatures = testFeatures*P(:,1:input.dimPCA)';
        end
        if((strcmpi(input.typeDA, 'gfk') || strcmpi(input.typeDA, 'SA')) && ~isempty(testFeatures))
            testFeatures = testFeatures*W;
        end
        
    end
    
    % Training of discriminative classifiers
    if(isempty(transferLabels))
        if(isempty(out_classifiers) && ~(input.isDA && strcmpi(input.typeDA,'corr')))
            if(strfind(input.typeClassifier,'SVM'))
                out_classifiers = trainLabels(input, srcData, srcFeatures, srcIds, srcClasses, input.typeClassifier);
            elseif(strcmpi(input.typeClassifier,'kNN'))
                transferLabels = transferKNN(srcFeatures, srcIds, srcClasses, tgtData, tgtFeatures);
            end
        end

        if(~strcmpi(input.typeClassifier,'kNN') && (~input.isDA || ~(input.isDA && strcmpi(input.typeDA,'corr'))))
            [transferLabels, scores] = assignLabels(input, out_classifiers, srcClasses, metadata, tgtIds, tgtFeatures);
        else
            scores = zeros(length(tgtData.annotations.imgId), size(srcClasses,1));
        end
    end

    if(input.isRebuttal && input.isWSVM)
        isBg = cellfun(@isempty,transferLabels);
        transferLabels(isBg) = repmat({'background'}, [sum(isBg) 1]);
    end
    
%     if(input.isRebuttal && ~input.isOpenset)
%         for i = 11:length(input.sourceDataset.classes)
%             isClass = ismember(transferLabels,input.sourceDataset.classes{i});
%             transferLabels(isClass) = repmat({'background'}, [sum(isClass) 1]);
%             isClass = ismember(tgtData.annotations.classes,input.sourceDataset.classes{i});
%             tgtData.annotations.classes(isClass) = repmat({'background'}, [sum(isClass) 1]);
%         end
%         input.sourceDataset.classes = [input.sourceDataset.classes(1:10), {'background'}];
%         input.targetDataset.classes = [input.sourceDataset.classes(1:10), {'background'}];
%         [srcIds, srcClasses, metadata] = getIdLabels(input.sourceDataset, srcData.annotations);
%         [tgtIds, tgtClasses] = getIdLabels(input.targetDataset, tgtData.annotations, metadata);
%     end
    
    out_transfer = transferLabels;
    %%%%%%% RESULTS %%%%%%%
    % - Confusion matrix and text file if label ground truth is provided
    if(size(tgtClasses,1) > 1)

        [path, name] = getResultsPath(input);
        path = [path '/' name];
        
        % Store tSNE visualisation (adapted) src + (gt) tgt
        if(input.isSaveTSNE)
            minDims = round(size(srcFeatures,2)*0.20);
            coeff = mPCA([srcFeatures; tgtFeatures]);
            sFeat = srcFeatures; %*coeff(:,1:minDims);
            tFeat = tgtFeatures; %*coeff(:,1:minDims);
            saveTSNE(input, getResultsPath(input), sFeat, srcIds, metadata, tFeat, tgtIds);
        end

        % Evalutation of accuracy:
        % SOURCE -> TARGET (transfer evaluation)
        confusionMatrix(input, metadata, srcClasses, tgtData, tgtClasses, tgtIds, transferLabels, path, '(SRC-TGT) ');
        % No more need for 4view supervision, others with test data only
        actual4View = input.is4ViewSupervised;
        input.is4ViewSupervised = false;
        input.isClassSupervised = false;
        
        % SOURCE -> TEST (comparison src, direct test data labelling)
        if(0) % ~isempty(testFeatures)) % && (strcmpi(input.typeDA,'FMO') || strcmpi(input.typeDA,'corr')))
            if(isempty(out_classifiers))
                out_classifiers = trainLabels(input, srcData, srcFeatures, srcIds, srcClasses, input.typeClassifier);
            end
%             if(strcmpi(input.trainDomain,'src') || strcmpi(input.typePipeline,'class'))
                testLabels = assignLabels(input, out_classifiers, srcClasses, metadata, testIds, testFeatures);
                confusionMatrix(input, metadata, srcClasses, testData, tgtClasses, testIds, testLabels, path, '(SRC-TEST) ');
%             end

            if(~strcmpi(class(input.sourceDataset), class(input.targetDataset)) || isprop(input.sourceDataset,'source'))

                if(1) % strcmpi(input.trainDomain,'tgt') && strcmpi(input.typePipeline,'class'))
                    % TGT -> TEST (comparison tgt - training target relabelled)
                    classifiers = trainLabels(input, tgtData, tgtFeatures, transferLabels, srcClasses, input.typeClassifier);
                    if(strcmpi(input.trainDomain,'tgt'))
                        out_classifiers = classifiers;
                    end
                    testLabels = assignLabels(input, classifiers, srcClasses, metadata, testIds, testFeatures);
                    confusionMatrix(input, metadata, srcClasses, testData, tgtClasses, testIds, testLabels, path, '(TGT-TEST) ');
                end
                if(~strcmpi(input.typeDA,'corr'))
                    if(~input.isDA) % strcmpi(input.trainDomain,'tgt_gt') && strcmpi(input.typePipeline,'class'))
                        % TGT_GT -> TEST (comparison gt - training target gt)
                        out_transfer = tgtIds; % no actual transfer, use of gt data!
                        classifiers = trainLabels(input, tgtData, tgtFeatures, tgtIds, srcClasses, input.typeClassifier);
                        if(strcmpi(input.trainDomain,'tgt_gt'))
                            out_classifiers = classifiers;
                        end
                        testLabels = assignLabels(input, classifiers, srcClasses, metadata, testIds, testFeatures);
                        confusionMatrix(input, metadata, srcClasses, testData, tgtClasses, testIds, testLabels, path, '(TGT_GT-TEST) ');
                    end

                    if(strcmpi(input.typePipeline,'class'))
                        if(1) % strcmpi(input.trainDomain,'tgt'))
                            % JOINT SRC+TGT -> TEST (comparison joint: src+tgt relabelled)
                            classifiers = trainLabels(input, srcData, [srcFeatures; tgtFeatures], [srcIds; transferLabels], srcClasses, input.typeClassifier);
                            if(strcmpi(input.trainDomain,'both'))
                                out_classifiers = classifiers;
                            end
                            testLabels = assignLabels(input, classifiers, srcClasses, metadata, testIds, testFeatures);
                            confusionMatrix(input, metadata, srcClasses, testData, tgtClasses, testIds, testLabels, path, '(JOINT-TEST) ');
                        elseif(strcmpi(input.trainDomain,'tgt_gt'))
                            % JOINT SRC+TGT_GT -> TEST (comparison joint: src+tgt_gt)
                            classifiers = trainLabels(input, srcData, [srcFeatures; tgtFeatures], [srcIds; tgtIds], srcClasses, input.typeClassifier);
                            testLabels = assignLabels(input, classifiers, srcClasses, metadata, testIds, testFeatures);
                            confusionMatrix(input, metadata, srcClasses, testData, tgtClasses, testIds, testLabels, path, '(JOINT_GT-TEST) ');
                        end
                    end
                end
            end
        end

        input.is4ViewSupervised = actual4View;
        input.isClassSupervised = actual4View;
        
    end

    % - Save training images with label information
    if(input.isStoreTransferOutput)
        % fprintf('Saving target images with their transfers (classifications)...\n');
        % saveImagesLabelTransfer(input, metadata, tgtData, transferLabels, tgtIds, scores);
        saveLabelTransfer(input, metadata, transferLabels);
    end
    
end
