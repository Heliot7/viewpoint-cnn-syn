# Z:/PhD/Data/CNN/VP_VGG_all Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 30 70000 class 0 15 0 16 12
# Z:/PhD/Data/CNN/VP_VGG_all Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all Z:/PhD/Data/Syn/o3d_1000/lmdb_0/all "" 224 30 70000 class 0 15 0 16 12

# 21_baseline (no scaling)
# Z:/PhD/Data/CNN/VP_VGG_all/21_baseline Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 30 70000 class 0 17.143 0 16 12

# 15_warped (scaling)
# Z:/PhD/Data/CNN/VP_VGG_all/15_warped Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 30 70000 class 0 15 0 16 12

# 15_ar (scaling)
# Z:/PhD/Data/CNN/VP_VGG_all/15_ar Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 30 70000 class 0 15 1 16 12

# 5_ar (scaling)
# Z:/PhD/Data/CNN/VP_VGG_all/5_ar Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 30 70000 class 0 5 1 16 12

# 1_ar (scaling)
# Z:/PhD/Data/CNN/VP_VGG_all/1_ar Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 30 70000 class 0 1 1 16 12

# Z:/PhD/Data/CNN/VP/15_ar/vp_class_all_ReSh Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all_val "" Z:/PhD/Data/SynShapeNet/Pascal3D/lmdb_0/all 224 24 100000 class 0 15 1 16 12

import sys
import os
import math
import caffe
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayers(data_real, data_test, data_syn, data_shapenet, batch_size, batch_size_test, layername, kernel, stride,
              outCH, trans_param, trans_param_shape, patchSize, onlyAZ, binSize, deploy=False):

    n = caffe.NetSpec()
    assert len(layername) == len(kernel)
    assert len(layername) == len(stride)
    assert len(layername) == len(outCH)

    # Produce data definition for deploy net
    if deploy == False:

        # Train data
        nr = caffe.NetSpec()
        suffix_data = ''
        div_kp = 1.0
        div_vp = 1.0
        if data_real != '' and data_syn != '':
            suffix_data = '_real'
            div_kp = 2.0
        if (data_shapenet != '' and data_real != '') or (data_shapenet != '' and data_syn!= ''):
            suffix_data = '_real'
            div_vp = 2.0

        numDatasets = 0
        # -> Real data
        if data_real != '':
            nr.tops['data' + suffix_data], nr.tops['label_kp' + suffix_data], nr.tops['label_kp_pos' + suffix_data], \
            nr.tops['label_vp' + suffix_data], nr.tops['label_class' + suffix_data] = \
                L.PoseData(cpmdata_param=dict(backend=1, source=data_real,
                                              batch_size=int(math.floor(batch_size/div_kp/div_vp))),
                           transform_param=trans_param, include=dict(phase=caffe.TRAIN), ntop=5)
            data1 = suffix_data
            numDatasets += 1

        # -> Synthetic data
        if data_syn != '':
            suffix_data = '_syn'
            if data_real == '' and data_shapenet == '':
                suffix_data = ''
            nr.tops['data' + suffix_data], nr.tops['label_kp' + suffix_data], nr.tops['label_kp_pos' + suffix_data], \
            nr.tops['label_vp' + suffix_data], nr.tops['label_class' + suffix_data] = \
                L.PoseData(cpmdata_param=dict(backend=1, source=data_syn,
                                              batch_size=int(math.ceil(batch_size/div_kp/div_vp))),
                           transform_param=trans_param, include=dict(phase=caffe.TRAIN), ntop=5)
            if numDatasets > 0:
                data2 = suffix_data
            else:
                data1 = suffix_data
            numDatasets += 1

        # -> ShapeNet data
        if data_shapenet != ''  and binSize > 0:
            suffix_data = '_shape'
            if data_real == '' and data_syn == '':
                suffix_data = ''
            nr.tops['data' + suffix_data], nr.tops['label_vp' + suffix_data], nr.tops['label_class' + suffix_data] = \
                L.ViewpointData(cpmdata_param=dict(backend=1, source=data_shapenet, batch_size=int(batch_size/div_vp)),
                                transform_param=trans_param_shape, include=dict(phase=caffe.TRAIN), ntop=3)
            #if data_real != '' or data_syn != '':
            #    nr.data = L.Concat(nr.data_real, nr.data_syn, concat_param=dict(concat_dim=0),
            #                       include=dict(phase=caffe.TRAIN))
            if numDatasets == 1:
                data2 = suffix_data
            elif numDatasets == 2:
                data3 = suffix_data
            numDatasets += 1

        # Concatenate Train data
        if numDatasets == 2:
            nr.data = L.Concat(nr.tops['data' + data1], nr.tops['data' + data2],
                               concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            if data_shapenet == '':
                nr.label_kp = L.Concat(nr.tops['label_kp' + data1], nr.tops['label_kp' + data2],
                                       concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
                nr.label_kp_pos = L.Concat(nr.tops['label_kp_pos' + data1], nr.tops['label_kp_pos' + data2],
                                           concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            else:
                nr.label_kp = L.Concat(nr.tops['label_kp' + data1],
                                       concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
                nr.label_kp_pos = L.Concat(nr.tops['label_kp_pos' + data1],
                                           concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            nr.label_vp = L.Concat(nr.tops['label_vp' + data1], nr.tops['label_vp' + data2],
                                   concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            nr.label_class = L.Concat(nr.tops['label_class' + data1], nr.tops['label_class' + data2],
                                      concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
        if numDatasets == 3:
            nr.data = L.Concat(nr.tops['data' + data1], nr.tops['data' + data2], nr.tops['data' + data3],
                               concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            nr.label_kp = L.Concat(nr.tops['label_kp' + data1], nr.tops['label_kp' + data2],
                                   concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            nr.label_kp_pos = L.Concat(nr.tops['label_kp_pos' + data1], nr.tops['label_kp_pos' + data2],
                                       concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            nr.label_vp = L.Concat(nr.tops['label_vp' + data1], nr.tops['label_vp' + data2],
                                   nr.tops['label_vp' + data3], concat_param=dict(concat_dim=0),
                                   include=dict(phase=caffe.TRAIN))
            nr.label_class = L.Concat(nr.tops['label_class' + data1], nr.tops['label_class' + data2],
                                      nr.tops['label_class' + data3], concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))

        # Storage of training data
        train_data = str(nr.to_proto())

        # Test data
        n.data, n.tops['label_kp'], n.tops['label_kp_pos'], n.tops['label_vp'], n.tops['label_class'] = \
            L.PoseData(name='data', cpmdata_param=dict(backend=1, source=data_test, batch_size=batch_size_test),
                       transform_param=trans_param, include=dict(phase=caffe.TEST), ntop=5)
        if data_real != '' or data_syn != '':
            n.Silence_kp = L.Silence(n.tops['label_kp'], ntop=0)
            n.Silence_kp_pos = L.Silence(n.tops['label_kp_pos'], ntop=0)
        else:
            n.Silence_kp = L.Silence(n.tops['label_kp'], include=dict(phase=caffe.TEST), ntop=0)
            n.Silence_kp_pos = L.Silence(n.tops['label_kp_pos'], include=dict(phase=caffe.TEST), ntop=0)
    else:
        n.data = L.Input(input_param=dict(shape=dict(dim=[1, 3, patchSize, patchSize])),
                         transform_param=trans_param, ntop=1)

    # just follow arrays..CPCPCPCPCCCC....
    last_layer = 'data'
    layer_counter = 1
    conv_counter = 0

    for l in range(0, len(layername)):
        if layername[l] == 'C':
            conv_counter += 1
            conv_name = 'conv%d_%d' % (layer_counter, conv_counter)
            n.tops[conv_name] = L.Convolution(n.tops[last_layer], kernel_size=kernel[l],
                                                  num_output=outCH[l], pad=int(math.floor(kernel[l]/2))) #,
                                                  # param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
            last_layer = conv_name
        elif layername[l] == 'A':  # ReLu
            if layer_counter < 6:
                ReLUname = 'relu%d_%d' % (layer_counter, conv_counter)
            else:
                ReLUname = 'relu%d' % (layer_counter)
            n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
            last_layer = ReLUname
        elif layername[l] == 'P': # Pooling
            pool_name = 'pool%d' % (layer_counter)
            n.tops[pool_name] = L.Pooling(n.tops[last_layer], kernel_size=kernel[l], stride=stride[l], pool=P.Pooling.MAX)
            last_layer = pool_name
            layer_counter += 1
            conv_counter = 0
        elif layername[l] == 'D':
            drop_name = 'drop%d' % (layer_counter)
            n.tops[drop_name] = L.Dropout(n.tops[last_layer], in_place=True, dropout_param=dict(dropout_ratio=0.5))
            layer_counter += 1
            last_layer = drop_name
        elif layername[l] == 'F':
            fc_name = 'fc%d' % layer_counter
            n.tops[fc_name] = L.InnerProduct(n.tops[last_layer], inner_product_param=dict(num_output=outCH[l])) # param=[dict(lr_mult=1), dict(lr_mult=2)])
            last_layer = fc_name
        # Add FC layer
        elif layername[l] == 'K' or layername[l] == 'R':
            n.tops['fc1_stage_vp1'] = L.InnerProduct(n.tops['fc7'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                     inner_product_param=dict(num_output=outCH[l]))
            # + Loss layer (and Accuracy if classification task)
            if deploy == False:
                if layername[l] == 'K':
                    n.tops['mask_fc1_stage_vp1'] = L.MaskOutputs(n.tops['fc1_stage_vp1'], n.tops['label_class'],
                                                                 mask_outputs_param=dict(kernel_size=outCH[l] / numClasses))
                    if onlyAZ:
                        n.tops['loss_stage_vp_az1'] = L.SoftmaxWithLoss(n.tops['mask_fc1_stage_vp1'], n.tops['label_vp'])
                        n.tops['accuracy_vp_az1'] = L.Accuracy(n.tops['fc1_stage_vp1'],n.tops['label_vp'],
                                                                            include=dict(phase=caffe.TEST))
                    else:
                        n.tops['fc1_stage_vp_az1'], n.tops['fc1_stage_vp_el1'], n.tops['fc1_stage_vp_th1'] = \
                            L.Slice(n.tops['mask_fc1_stage_vp1'],
                                    slice_param=dict(axis=1, slice_point=[int(round(360 / binSize)),
                                                                          int(round(360 / binSize) + (round(180 / binSize) + 1))]), ntop=3)
                        n.tops['label_vp_az'], n.tops['label_vp_el'], n.tops['label_vp_th'] = L.Slice(n.tops['label_vp'], slice_param=dict(axis=1,slice_point=[1,2]),ntop=3)
                        n.tops['loss_stage_vp_az1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_az1'],n.tops['label_vp_az'])
                        n.tops['accuracy_vp_az1'] = L.Accuracy(n.tops['fc1_stage_vp_az1'],n.tops['label_vp_az'],include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_el1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_el1'],n.tops['label_vp_el'])
                        n.tops['accuracy_vp_el1'] = L.Accuracy(n.tops['fc1_stage_vp_el1'],n.tops['label_vp_el'],include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_th1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_th1'],n.tops['label_vp_th'])
                        n.tops['accuracy_vp_th1'] = L.Accuracy(n.tops['fc1_stage_vp_th1'],n.tops['label_vp_th'],include=dict(phase=caffe.TEST))
                elif layername[l] == 'R':
                    n.tops['mask_fc1_stage_vp1'] = L.MaskOutputs(n.tops['fc1_stage_vp1'], n.tops['label_class'],
                                                                 mask_outputs_param=dict(kernel_size=outCH[l]/numClasses))
                    n.tops['loss_stage_vp1'] = L.SmoothL1Loss(n.tops['mask_fc1_stage_vp1'],
                                                              n.tops['label_vp'], loss_weight=1)

    if deploy == False:
        return train_data + str(n.to_proto())
    else:
        return 'name: "VP"\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[1:])


def writePrototxts(dataFolder, test_source, synDataFolder, shapeNetDataFolder, dir, batch_size, batch_size_test,
                   num_iter_cnn, layername, kernel, stride, outCH, trans_param, trans_param_shapenet, base_lr, step_lr,
                   patchSize, onlyAZ, binSize):

    # Create folder object class
    # write the net prototxt files out
    with open('%s/all_train_test.prototxt' % dir, 'w') as f:
        print 'writing %s/all_train_test.prototxt' % dir
        str_to_write = setLayers(dataFolder, test_source, synDataFolder, shapeNetDataFolder, batch_size,
                                 batch_size_test, layername, kernel, stride, outCH, trans_param, trans_param_shapenet,
                                 patchSize, onlyAZ, binSize, deploy=False)
        f.write(str_to_write)

    with open('%s/all_deploy.prototxt' % dir, 'w') as f:
         print 'writing %s/all_deploy.prototxt' % dir
         str_to_write = setLayers(dataFolder, test_source, synDataFolder, shapeNetDataFolder, batch_size,
                                  batch_size_test, layername, kernel, stride, outCH, trans_param, trans_param_shapenet,
                                  patchSize, onlyAZ, binSize, deploy=True)
         f.write(str_to_write)

    if not os.path.exists(directory):
         os.makedirs(directory)
    with open('%s/all_solver.prototxt' % dir, "w") as f:
         solver_string = getSolverPrototxt(dir, num_iter_cnn, base_lr, step_lr)
         print 'writing %s/all_solver.prototxt' % dir
         f.write('%s' % solver_string)


def getSolverPrototxt(proto_folder, num_iter_cnn, base_lr, step_lr):
    string = 'net: "%s/all_train_test.prototxt"\n\
# The base learning rate, momentum and the weight decay of the network.\n\
# Evaluation of test data\n\
test_iter: 250\n\
test_interval: 10000\n\
base_lr: %f\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
# The learning rate policy\n\
lr_policy: "step"\n\
gamma: 0.1\n\
stepsize: %d\n\
# Display every 100 iterations\n\
display: 2500\n\
# The maximum number of iterations\n\
max_iter: %d\n\
# snapshot intermediate results\n\
snapshot: %d\n\
snapshot_prefix: "%s/all"\n\
# solver mode: CPU or GPU\n\
solver_mode: GPU\n' % (proto_folder, base_lr, step_lr, num_iter_cnn, num_iter_cnn, proto_folder)
    return string


if __name__ == "__main__":

    # sys.argv[1]: folder prototxt folder (solver and model together + caffe weights to store)
    print "prototxt folder: " + sys.argv[1]
    # sys.argv[2]: lmdb folder (real)
    print "lmdb real folder: " + sys.argv[2]
    # sys.argv[3]: lmdb folder (real val)
    print "lmdb real folder: " + sys.argv[3]
    # sys.argv[4]: lmdb folder (syn)
    print "lmdb syn folder: " + sys.argv[4]
    # sys.argv[5]: lmdb shapenet folder
    print "lmdb shapenet folder: " + sys.argv[5]
    # sys.argv[6]: size patch
    print "size patch: [" + sys.argv[6] + "," + sys.argv[6] + "]"
    # sys.argv[7]: size batch
    print "size batch: " + sys.argv[7]
    # sys.argv[8]: size batch test
    print "size batch test: " + sys.argv[8]
    # sys.argv[9]: number of iterations CNN
    print "num iter CNN: " + sys.argv[9]
    # sys.argv[10]: learning rate CNN
    print "learning rate: " + sys.argv[10]
    # sys.argv[11]: stepsize CNN
    print "step size: " + sys.argv[11]
    # sys.argv[12]: type viewpoint estimation
    print "type viewpoint estimation: " + sys.argv[12]
    # sys.argv[13]: if only azimuth viewpoint
    print "if only azimuth viewpoint: " + sys.argv[13]
    # sys.argv[14]: only VP information or also with Parts
    print "viewpoint bin size: " + sys.argv[14]
    # sys.argv[15]: keep or not the AR when resizing to crop_size
    print "keep AR?: " + sys.argv[15]
    # sys.argv[16]: number of padded pixels in output image
    print "Padded pixels: " + sys.argv[16]
    # sys.argv[17]: number of classes
    print "number classes: " + sys.argv[17]

    ### Change here for different dataset
    directory = sys.argv[1] + '/all'
    dataFolder = sys.argv[2]
    testFolder = sys.argv[3] # dataFolder + '_val'
    synDataFolder = sys.argv[4]
    dataShapeNetFolder = sys.argv[5]
    patchSize = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    batch_size_test = int(sys.argv[8])
    num_iter_cnn = int(sys.argv[9])
    base_lr = float(sys.argv[10])
    step_lr = int(sys.argv[11])
    typeVP = sys.argv[12]
    onlyAZ = bool(int(sys.argv[13]))
    binSize = float(sys.argv[14])
    keepAR = bool(int(sys.argv[15]))
    pad = int(sys.argv[16])
    numClasses = int(sys.argv[17])
    ###

    if not os.path.exists(directory):
        os.makedirs(directory)

    transform_param = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize, scale_prob=0, scale_min=0.6,
                           scale_max=1.0, max_rotate_degree=0, center_perterb_max=0.9, put_gaussian=False,
                           flip_prob=0.5, is_rigid=True, type_vp=typeVP, only_azimuth=onlyAZ, keep_ar=keepAR, pad=pad,
                           size_bin_vp=binSize)
    transform_param_shapenet = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize, scale_prob=0, scale_min=0.6,
                                    scale_max=1.0, max_rotate_degree=0, center_perterb_max=0.9, put_gaussian=False,
                                    flip_prob=0.5, is_rigid=True, type_vp=typeVP, only_azimuth=onlyAZ, keep_ar=keepAR,
                                    pad=pad, size_bin_vp=binSize)

    layername = ['C', 'A', 'C', 'A', 'P', # conv1
                 'C', 'A', 'C', 'A', 'P', # conv2
                 'C', 'A', 'C', 'A', 'C', 'A', 'P', # conv3
                 'C', 'A', 'C', 'A', 'C', 'A', 'P', # conv4
                 'C', 'A', 'C', 'A', 'C', 'A', 'P', # conv5
                 'F', 'A', 'D', # fc6
                 'F', 'A', 'D'] # fc7
    kernel = [3, 0, 3, 0, 2,
              3, 0, 3, 0, 2,
              3, 0, 3, 0, 3, 0, 2,
              3, 0, 3, 0, 3, 0, 2,
              3, 0, 3, 0, 3, 0, 2,
              0, 0, 0,
              0, 0, 0]
    outCH = [64, 0, 64, 0, 0,
             128, 0, 128, 0, 0,
             256, 0, 256, 0, 256, 0, 0,
             512, 0, 512, 0, 512, 0, 0,
             512, 0, 512, 0, 512, 0, 0,
             4096, 0, 0,
             4096, 0, 0]
    stride = [0, 0, 0, 0, 2,
              0, 0, 0, 0, 2,
              0, 0, 0, 0, 0, 0, 2,
              0, 0, 0, 0, 0, 0, 2,
              0, 0, 0, 0, 0, 0, 2,
              0, 0, 0,
              0, 0, 0]
    names = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']

    if typeVP == "class":
        layername += ['K']
        kernel += [0]
        stride += [0]
        if onlyAZ:
            outCH += [int(numClasses * round(360 / binSize))]
        else:
            outCH += [int(numClasses * round(360 / binSize)) +
                      int(numClasses * (round(180 / binSize) + 1)) +
                      int(numClasses * round(360 / binSize))]
    elif typeVP == "reg":
        layername += ['R']
        kernel += [0]
        stride += [0]
        if onlyAZ:
            outCH += [numClasses*2]
        else:
            outCH += [numClasses*6]

    writePrototxts(dataFolder, testFolder, synDataFolder, dataShapeNetFolder, directory, batch_size, batch_size_test,
                   num_iter_cnn, layername, kernel, stride, outCH, transform_param, transform_param_shapenet,
                   base_lr, step_lr, patchSize, onlyAZ, binSize)