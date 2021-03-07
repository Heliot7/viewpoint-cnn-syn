% Class file with all input parameters.
classdef InputParameters < dynamicprops
    properties        
        %% Pipeline and samples %%
        % - I/O paths
        PATH_DATA = 'Z:\PhD\Data\';
        PATH_RESULTS = 'Z:\PhD\Results\Results_O3D\';
        % Type of object recognition task
        typePipeline = 'class'; % ['class', 'det', 'classDet', 'pose']
        % - Training per class (+Inf means that all images are included)
        numSrcTrain = +Inf;
        numTgtTrain = +Inf;
        numTgtTrainDA = +Inf;
        % - Number of test images in the detection phases
        numImages = +Inf;
        % - Output parameters
        testValidation = false;
        isSaveTSNE = false;
        isStoreTransferOutput = true;
        isStoreImgOutputPre = true;
        isStoreImgOutputPost = true;
        isShowPose = false;
        % - Random seed for tests (selected images, candidates...)
        seedRand = 1;
        
        %% Datasets %%
        sourceDataset = Synthetic; % ObjectNet3D
        targetDataset = ObjectNet3D; % Pascal3D
        % human: [Synthetic, CVC, INRIA, Daimler, TUD, Caltech]
        % car: [Synthetic, ObjCat3D, EPFL, KITTI, Pascal3D, NYC3DCARS]
        % multi: [Saenko, Office, CrossDataset_dense]
        % - whether BB patch size is automatically computed based on training samples
        autoBB = false;
        isScaleInvariance = false;
        
        % Feature descriptors
        typeDescriptor = 'HOG'; % ['HOG', 'BoW', 'CNN-fc7', 'CNN-conv5', 'CNN-conv3']
        keepAR = false;
        isMirrored = false;
        isZScore = true;
        % - CNN 
        PATH_CAFFE = 'D:\Core\Caffe-git\';
        PATH_CNN = 'Z:\PhD\Data\CNN\';
        cnnName = 'AlexNet'; % ['AlexNet', 'CaffeNet', 'GoogleNet', 'VGG', 'ResNet']
        cnnModel = 'AlexNet'; % ['AlexNet', 'FT_AlexNet', 'finetune_voc_2012']
        cnnPoseTypeCNN = ''; % 'class', 'reg', 'map'
        
        %% Classifiers %%
        typeClassifier = 'LSVM'; % ['LSVM', 'kNN', 'CNN']
        % - Precondition in supervision: 'class' attribute known
        isClassSupervised = false;
        is4ViewSupervised = true;
        % - LSVM
        methodSVM = 'libsvm' % 'libsvm' (LSVM & SVM), 'liblinear' (LSVM)
        multiClassApproach = 'OVA'; % 'OVA': One-vs-All, 'OVO': One-vs-One
        numIterLSVM = 100;
        C_LSVM = 0.001;
        CV_LSVM = false;
        
        %% Domain Adaptation %%
        isDA = false;
        typeDA = 'FMO'; % ['FMO', 'gfk', SA', 'MMDT', 'CORAL', ...}
        % 'whitening', 'gfk', 'MMDT', 'shekhar', 'saenko', 'DASVM'
        % - class-based problems
        daAllSrc = true; % true, takes all source samples
        daAllTgt = true; % true, takes all target samples
        daOnlySupervised = false; % true, test data is embedded (label 0)
        daNumSupervised = 3;
        % - FMO attributes
        iterResetDA = 1;
        iterDA = 1;
        numIterFMO = 1;
        numIterOpt = 50;
        regParam = 0.0;
        transformationDomain = 'src';
        dimPCA = 0;
        numTgtClusters = 99999;
        numSrcClusters = 16;
        deltaW = 1.0;
        numCorr = 1; % extra source for unbalances [1..Inf]*(numTgt/numSrc)
        numLambda = 1.0; % distance samples empty nodes <-> tgt samples
        tol_residual = 0.01;
        tol_W = 0.0;
        %% LC
        numNN = 0;
        isClosestNN = false;
        dynLambda = false;
        initDynLambda = 0.8;
        limDynLambda = 0.5;
        stepDynLambda = 0.1;
        isLocalSrc = false;
        isFMOAllSamples = true;
        %% Background handle
        % - Include Bg samples in correspondence estimation
        includeBgClass = true;
        % - Ignore unknown classe
        isWild = false;
        % - Type of supervision (Office or 'Number')
        typeWildSupervision = '100';
        % Print-outs
        isMidResultsDA = false;
        isDaView2D = false;
        % Special test cases 
        daSpecial = ''; % {'', 'lsvm', 'gt', 'rnd' };
        
        %% Pose Estimation %%
        goCubicUpscale = true;
        
        %% Object detection %%
        trainDomain = 'tgt'; % ['src', 'tgt', 'both']
        jointDetPose = true;
        % - Negatives
        numTrainNeg = 25000; % - Need to be an even number
        numValidationHardNegatives = 250;
        numHardNegatives = 1000;
        numIterHardNegatives = 0;
        % - Multiresolution pyramid
        multiResTest = 1.0:0.25:2.0;
        % - Random negative samples overlap
        lvlOverlapNegSamples = 0.0;
        % - Non-maximum suppression overlap
        lvlOverlapMerge = 0.3;
        % - Evaluation overlap (PASCAL VOC)
        lvlOverlapTest = 0.5;
        % - 50% overlap
        detTh = 'VOC'; % ['VOC' (50% overlap PASCAL)]
        % - Detect using R-CNN methodology
        isSelectiveSearch = false;
    end

end