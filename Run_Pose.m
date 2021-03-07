input = InputParameters;
% input.isShowPose = true;
input.PATH_RESULTS = '<path>\Results_O3D\';
input.jointDetPose = true;
input.goCubicUpscale = true;
input.cnnPoseTypeCNN = 'class';
input.cnnName = 'CPM-VP\O3D\cpm-vp_class_all_Re';
input.cnnModel = 'iter_150000';
main_pose_test(input,'ObjectNet3D');
