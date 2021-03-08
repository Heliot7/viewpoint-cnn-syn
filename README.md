# Joint Viewpoint & Keypoint estimation with CNNs (Caffe)

Paper (oral) GCPR 2019: https://arxiv.org/pdf/1912.06274.pdf
supp. material: http://pages.iai.uni-bonn.de/gall_juergen/download/kp-vpi_supp.pdf

##### Models
Models Keypoint + Viewpont Estimation (Caffe):

- ObjectNet3D - KP+VP Real Data https://drive.google.com/file/d/1ZdF6RZUkUrtf7yfvnt8L-_CmK58nyvv7/view?usp=sharing
- ObjectNet3D - KP+VP Real + Synthetic (ShapeNet+Own) Data
https://drive.google.com/file/d/11Njc2Z8W3jzQg8428LXj1unIjLFtfFOZ/view?usp=sharing

Models Viewpoint Estimation Only (Caffe):
- ObjectNet3D - Classication Real Data
https://drive.google.com/file/d/1BdUan6iMjUm-o4m8gLKK71RWwNTPYJMy/view?usp=sharing
- ObjectNet3D - Classification Real + Synthetic (ShapeNet+Own) Data
https://drive.google.com/file/d/1WrJzm5MWVpdvVgutbQ-zOMrWa2TJys5W/view?usp=sharing
- Pascal3D - Classication Real Data
https://drive.google.com/file/d/13iu9UOGRIEOtZPV3-RLHXGEHiFXlDgFA/view?usp=sharing
- Pascal3D - Classification Real + Synthetic (ShapeNet+Own) Data
https://drive.google.com/file/d/1VLKK_mRmVn3tYygJBwmBEikdpeaxeXUK/view?usp=sharing

##### How to set-up the CNN framework
- Caffe-git is an enhanced copy of Caffe version github on Dec 2016
Source: https://drive.google.com/file/d/1FjJcTY8x4lxyk_ErZIto9O52uaD62OsR/view?usp=sharing
- The given VS2013 should be working straight forward
Binaries: https://drive.google.com/file/d/1IcQJGt6U_fxJzkD2q2lm9bqzmPwP1nUy/view?usp=sharing
- caffe.mexw64 for matlab found here: ~/features/CNN/

##### Requirements
- Matlab (tested on version 2013a)
- Cygwin terminal for running the trainings
- Python 2.7.1X

##### Setup, train and test models (Windows)
- Training: Create models to be trained (matlab):
  - have a look at main_pose_train.m, based on InputParameters.m, the core function
  - cpm = only keypoint estimation, vp = only viewpoint and cpm-vp = joint trainingTrain a model:
  - Open a cygwin (a cmd promp should work as well, although it might crashed right after the first training, in case you have a sequence of models to be trained in a .sh file. 
  - The generated models to be trained (solver.prototxt, traintest.prototxt will be called from "./train.sh".
- Test those models:
  - Follow Run_Pose  and modify the 2 main functions: main_pose_test.m and poseEstimation.m if necessary. They are also based on InputParameters.m

##### Usage of synthetic data
- "synthetic" data means synthetic data generated from Render repository
- "shapenet" data means synthetic data from the shapenet repository: https://github.com/ShapeNet/RenderForCNN

##### Datasets
- You can obtain the gt keypoints from Tulsiani et al. (ICCV 2015) from: http://www.cs.berkeley.edu/~shubhtuls/cachedir/vpsKps/segkps.zip
- Caffe pre-trained model weights can be downloaded here:
  - VGG (16 layers): http://www.robots.ox.ac.uk/~vgg/research/very_deep/
  - Human keypoint (pose) estimation: http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_MPI/pose_iter_320000.caffemodel
  
