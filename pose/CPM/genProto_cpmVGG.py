# all Z:/PhD/Data/CNN/CPM_VGG_all Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 25 150000 0.00003 50000 12 17 6 1 8 class 0 0
# all Z:/PhD/Data/CNN/GlobalTest/cpm-vp_class_all_Re Z:/PhD/Data/Real/Multi/ObjectNet3D/lmdb_0/all "" "" 224 32 10 150000 0.0004 50000 12 17 6 1 16 class 0 15

import sys
import os
import caffe
import math
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayers(data_real, data_test, data_syn, data_shapenet, trans_param_train, trans_param_test, trans_param_shapenet,
              batch_size_train, batch_size_test, layername, kernel, stride, outCH, label_name, patchSize, numClasses,
              numParts, numStages, typeVP, onlyAZ, binSize, deploy=False):

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
                                              batch_size=int(math.floor(batch_size_train/div_kp/div_vp))),
                           transform_param=trans_param_train, include=dict(phase=caffe.TRAIN), ntop=5)
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
                                              batch_size=int(math.ceil(batch_size_train/div_kp/div_vp))),
                           transform_param=trans_param_train, include=dict(phase=caffe.TRAIN), ntop=5)
            if numDatasets > 0:
                data2 = suffix_data
            else:
                data1 = suffix_data
            numDatasets += 1

        # -> ShapeNet data
        if data_shapenet != ''  and binSize >= 1:
            suffix_data = '_shape'
            if data_real == '' and data_syn == '':
                suffix_data = ''
            nr.tops['data' + suffix_data], nr.tops['label_vp' + suffix_data], nr.tops['label_class' + suffix_data] = \
                L.ViewpointData(cpmdata_param=dict(backend=1, source=data_shapenet, batch_size=int(batch_size_train/div_vp)),
                                transform_param=trans_param_shapenet, include=dict(phase=caffe.TRAIN), ntop=3)
            # if data_real != '' or data_syn == '':
            #   nr.data = L.Concat(nr.data_real, nr.data_syn, concat_param=dict(concat_dim=0),
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

        # Slice class labels of kp part (first half) for masking each class in loss
        if data_shapenet != '' and binSize >= 1 and numClasses > 1:
            nr.label_class_kps, nr.label_class_vp = L.Slice(nr.tops['label_class'], slice_param=dict(slice_dim=0),
                                                            include=dict(phase=caffe.TRAIN), ntop=2)
            nr.tops['kill_class_vp'] = L.Silence(nr.label_class_vp, include=dict(phase=caffe.TRAIN), ntop=0)
        # Storage of training data
        train_data = str(nr.to_proto())

        # Test data
        n.data, n.tops['label_kp'], n.tops['label_kp_pos'], n.tops['label_vp'], n.tops['label_class'] = \
            L.PoseData(name='data', cpmdata_param=dict(backend=1, source=data_test, batch_size=batch_size_test),
                       transform_param=trans_param_test, include=dict(phase=caffe.TEST), ntop=5)
        if numClasses == 1:
            n.Silence_kp_pos = L.Silence(n.tops['label_kp_pos'], ntop=0)
            n.Silence_class = L.Silence(n.tops['label_class'], ntop=0)
        if typeVP == '':
            n.Silence_vp = L.Silence(n.tops['label_vp'], ntop=0)
        # Slice class labels of kp part (first half) for masking each class in loss
        if data_shapenet != '' and binSize >= 1 and numClasses > 1:
            n.label_class_kps = L.Concat(n.tops['label_class'],
                                         concat_param=dict(concat_dim=0), include=dict(phase=caffe.TEST))
        n.tops[label_name[1]], n.tops[label_name[0]] = \
            L.Slice(n.tops['label_kp'], slice_param=dict(axis=1, slice_point=numParts), ntop=2)
    else:
        n.data = L.Input(input_param=dict(shape=dict(dim=[1, 4, patchSize, patchSize])),
                         transform_param=trans_param_test, ntop=1)

    # something special before everything
    n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
    n.pool_center_lower = L.Pooling(n.center_map, kernel_size=9, stride=8, pool=P.Pooling.AVE)

    # just follow arrays..CPCPCPCPCCCC....
    last_layer = 'image'
    stage = 1
    conv_counter = 1
    layer_counter = 1

    for l in range(0, len(layername)):
        if layername[l] == 'C':
            suffix_conv = ''
            if (layername[l + 1] == 'L' or layername[l - 1] == '@'):
                suffix_conv = '_FT'
            if stage == 1:
                lr_m = 1 # update to 4?
                if layer_counter < 4 or (layer_counter == 4 and conv_counter < 3):
                    conv_name = 'conv%d_%d' % (layer_counter, conv_counter)
                else:
                    conv_name = 'conv%d_%d_CPM%s' % (layer_counter, conv_counter, suffix_conv)
            else:
                conv_name = 'Mconv%d_stage%d%s' % (conv_counter, stage, suffix_conv)
                lr_m = 4
            n.tops[conv_name] = \
                L.Convolution(n.tops[last_layer], kernel_size=kernel[l],
                              num_output=outCH[l], pad=stride[l],
                              param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                              weight_filler=dict(type='gaussian', std=0.01),
                              bias_filler=dict(type='constant'))
            last_layer = conv_name
            conv_counter += 1
        elif layername[l] == 'R':  # ReLU
            if stage == 1:
                if layer_counter < 4 or (layer_counter == 4 and conv_counter-1 < 3):
                    ReLU_name = 'relu%d_%d' % (layer_counter, conv_counter-1)
                else:
                    ReLU_name = 'relu%d_%d_CPM' % (layer_counter, conv_counter-1)
            else:
                ReLU_name = 'Mrelu%d_stage%d' % (conv_counter-1, stage)
            n.tops[ReLU_name] = L.ReLU(n.tops[last_layer], in_place=True)
            last_layer = ReLU_name
            if layer_counter == 4 and conv_counter-1 == 7:
                layer_counter += 1
                conv_counter = 1
        elif layername[l] == 'P': # Pooling
            pool_name = 'pool%d_stage1' % (layer_counter)
            n.tops[pool_name] = L.Pooling(n.tops[last_layer], kernel_size=kernel[l], stride=stride[l], pool=P.Pooling.MAX)
            last_layer = pool_name
            layer_counter += 1
            conv_counter = 1
        elif layername[l] == 'D':
            if deploy == False:
                n.tops['drop%d_%d' % (layer_counter, conv_counter)] = L.Dropout(n.tops[last_layer], in_place=True, dropout_param=dict(dropout_ratio=0.2))
        elif layername[l] == '@':
            n.tops['concat_stage%d' % stage] = L.Concat(n.tops[last_layer], n.tops['conv4_7_CPM'], n.pool_center_lower, concat_param=dict(axis=1))
            last_layer = 'concat_stage%d' % stage
        elif layername[l] == 'L': # Loss: n.loss layer is only in training and testing nets, but not in deploy net.
            last_FT = last_layer
            if deploy == False:
                mask_name = last_layer
                label_class = 'label_class'
                if data_shapenet != '': # Slice between kps+vp and only vp (shapenet) datasets
                    mask_name = 'kps_features%d' % stage
                    n.tops[mask_name], n.tops['vp_features%d' % stage] = \
                        L.Slice(n.tops[last_layer], slice_param=dict(slice_dim=0),
                                include=dict(phase=caffe.TRAIN), ntop=2)
                    n.tops[mask_name] = L.Concat(n.tops[last_layer],
                                                 concat_param=dict(concat_dim=0), include=dict(phase=caffe.TEST))
                    n.tops['kill_vp%d' % stage] = L.Silence(n.tops['vp_features%d' % stage],
                                                            include=dict(phase=caffe.TRAIN), ntop=0)
                    last_layer = 'kps_features%d' % stage
                    label_class = 'label_class_kps'
                if numClasses > 1:
                    mask_name = 'mask_loss_stage%d' % stage
                    n.tops[mask_name] = L.MaskDynamic(n.tops[last_layer], n.tops[label_class], n.tops['label_kp_pos'],
                                                      mask_outputs_param=dict(kernel_size=numParts))
                if stage == 1:
                    n.tops['loss_stage%d' % stage] = \
                        L.EuclideanDynamicLoss(n.tops[mask_name], n.tops[label_name[0]], n.tops['label_kp_pos'])
                else:
                    n.tops['loss_stage%d' % stage] = \
                        L.EuclideanDynamicLoss(n.tops[mask_name], n.tops[label_name[1]], n.tops['label_kp_pos'])
            stage += 1
            conv_counter = 1
            last_layer = last_FT
        elif layername[l] == 'K' or layername[l] == 'G': # Add loss of viewpoint estimation after the last stage
            n.tops['fc1_stage_vp%d' % numStages] = L.InnerProduct(n.tops[last_layer],
                                                                  param=[dict(lr_mult=20, decay_mult=1),
                                                                         dict(lr_mult=40, decay_mult=0)],
                                                                  inner_product_param=dict(num_output=outCH[l]))
            last_layer = 'fc1_stage_vp%d' % numStages
            if deploy == False:
                if layername[l] == 'K':
                    if numClasses > 1:
                        mask_name = 'mask_fc1_stage_vp%d' % numStages
                        n.tops[mask_name] = L.MaskOutputs(n.tops[last_layer], n.tops['label_class'],
                                                          mask_outputs_param=dict(kernel_size=outCH[l]/numClasses))
                        last_layer = mask_name
                        layer_az = last_layer
                        label_az = 'label_vp'
                    if not onlyAZ:
                        layer_az = 'fc1_stage_vp_az%d' % numStages
                        n.tops[layer_az], \
                        n.tops['fc1_stage_vp_el%d' % numStages], \
                        n.tops['fc1_stage_vp_th%d' % numStages] = \
                            L.Slice(n.tops[last_layer],
                                    slice_param=dict(axis=1, slice_point=[int(round(360 / binSize)),
                                                                          int(round(360 / binSize)+(round(180 / binSize) + 1))]), ntop=3)
                        label_az = 'label_vp_az'
                        n.tops[label_az], n.tops['label_vp_el'], n.tops['label_vp_th'] = \
                            L.Slice(n.tops['label_vp'], slice_param=dict(axis=1, slice_point=[1, 2]), ntop=3)
                    n.tops['loss_stage_vp_az%d' % numStages] = L.SoftmaxWithLoss(n.tops[layer_az], n.tops[label_az])
                    n.tops['accuracy_vp_az%d' % numStages] = L.Accuracy(n.tops[layer_az], n.tops[label_az],
                                                                        include=dict(phase=caffe.TEST))
                    if not onlyAZ:
                        n.tops['loss_stage_vp_el%d' % numStages] = L.SoftmaxWithLoss(
                            n.tops['fc1_stage_vp_el%d' % numStages], n.tops['label_vp_el'])
                        n.tops['accuracy_vp_el%d' % numStages] = L.Accuracy(n.tops['fc1_stage_vp_el%d' % numStages],
                                                                            n.tops['label_vp_el'],
                                                                            include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_th%d' % numStages] = L.SoftmaxWithLoss(
                            n.tops['fc1_stage_vp_th%d' % numStages], n.tops['label_vp_th'])
                        n.tops['accuracy_vp_th%d' % numStages] = L.Accuracy(n.tops['fc1_stage_vp_th%d' % numStages],
                                                                            n.tops['label_vp_th'],
                                                                            include=dict(phase=caffe.TEST))
                elif layername[l] == 'G':
                    if numClasses > 1:
                        mask_name = 'mask_fc1_stage_vp%d' % numStages
                        n.tops[mask_name] = L.MaskOutputs(n.tops[last_layer], n.tops['label_class'],
                                                          mask_outputs_param=dict( kernel_size=outCH[l] / numClasses))
                        last_layer = mask_name
                    n.tops['loss_stage_vp%d' % numStages] = L.SmoothL1Loss(n.tops[last_layer],
                                                                           n.tops['label_vp'], loss_weight=1)

    # Final process
    allStrLayers = str(n.to_proto())
    for stage in range(1, numStages + 1):
        allStrLayers = allStrLayers.replace('Slice%d' % stage, 'kps_features%d' % stage)
    if deploy == False:
        return train_data + allStrLayers
    else:
        return 'name:"CPM_VGG"\n' + 'layer {' + 'layer {'.join(allStrLayers.split('layer {')[1:])


def writePrototxts(dataFolder, testFolder, dataSynFolder, dataShapeNetFolder, directory, batch_size_train,
                   batch_size_test, num_iter_cnn, base_lr, step_lr, layername, kernel, stride, outCH,
                   trans_param_train, trans_param_test,  trans_param_shapenet, task_name, label_name, patchSize,
                   numClasses, numParts, numStages, typeVP, onlyAZ, binSize):

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the net prototxt files out
    # > Train
    with open('%s/%s_train_test.prototxt' % (directory, task_name.lower()), 'w') as f:
        print 'writing %s/%s_train_test.prototxt' % (directory, task_name.lower())
        str_to_write = setLayers(dataFolder, testFolder, dataSynFolder, dataShapeNetFolder, trans_param_train,
                                 trans_param_test, trans_param_shapenet,  batch_size_train, batch_size_test, layername,
                                 kernel, stride, outCH, label_name, patchSize, numClasses, numParts, numStages, typeVP,
                                 onlyAZ, binSize, deploy=False)
        f.write(str_to_write)

    # > Deploy
    with open('%s/%s_deploy.prototxt' % (directory,task_name.lower()), 'w') as f:
        print 'writing %s/%s_deploy.prototxt' % (directory, task_name.lower())
        str_to_write = setLayers(dataFolder, testFolder, dataSynFolder, dataShapeNetFolder, trans_param_train,
                                 trans_param_test, trans_param_shapenet,  batch_size_train, batch_size_test, layername,
                                 kernel, stride, outCH, label_name, patchSize, numClasses, numParts, numStages, typeVP,
                                 onlyAZ, binSize, deploy=True)
        f.write(str_to_write)

    # Write the net solver
    with open('%s/%s_solver.prototxt' % (directory, task_name.lower()), "w") as f:
        solver_string = getSolverPrototxt(task_name, directory, num_iter_cnn, base_lr, step_lr)
        print 'writing %s/%s_solver.prototxt' % (directory, task_name.lower())
        f.write('%s' % solver_string)


def getSolverPrototxt(task_name, proto_folder, num_iter_cnn, base_lr, step_lr):
    num_snapshot = 10000 # num_iter_cnn
    string = 'net: "%s/%s_train_test.prototxt"\n\
# Evaluation of test data\n\
test_iter: 1000\n\
test_interval: 10000\n\
test_compute_loss: true\n\
# The base learning rate, momentum and the weight decay of the network.\n\
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
# Snapshot intermediate results\n\
snapshot: %d\n\
snapshot_prefix: "%s/%s"\n\
# Solver mode: CPU or GPU\n\
solver_mode: GPU\n' % (proto_folder, task_name, base_lr, step_lr, num_iter_cnn, num_snapshot, proto_folder, task_name)
    return string


if __name__ == "__main__":

    # sys.argv[1]: name task (class or "all")
    print "task: " + sys.argv[1]
    # sys.argv[2]: folder prototxt folder (solver and model together + caffe weights to store)
    print "prototxt folder: " + sys.argv[2]
    # sys.argv[3]: lmdb folder
    print "lmdb folder: " + sys.argv[3]
    # sys.argv[4]: lmdb folder (val)
    print "lmdb (val) folder: " + sys.argv[4]
    # sys.argv[5]: lmdb syn folder
    print "lmdb syn folder: " + sys.argv[5]
    # sys.argv[6]: lmdb shapenet folder
    print "lmdb shapenet folder: " + sys.argv[6]
    # sys.argv[7]: size patch
    print "size patch: [" + sys.argv[7] + "," + sys.argv[7] + "]"
    # sys.argv[8]: size batch train
    print "size batch train: " + sys.argv[8]
    # sys.argv[9]: size batch test
    print "size batch test: " + sys.argv[9]
    # sys.argv[10]: number of iterations CNN
    print "num iter CNN: " + sys.argv[10]
    # sys.argv[11]: learning rate CNN
    print "learning rate: " + sys.argv[11]
    # sys.argv[12]: stepsize CNN
    print "step size: " + sys.argv[12]
    # sys.argv[13]: number of classes
    print "number of classes: " + sys.argv[13]
    # sys.argv[14]: number of parts of largest Kps obj
    print "largest number of parts: " + sys.argv[14]
    # sys.argv[15]: sum of all object parts
    print "sum of parts: " + sys.argv[15]
    # sys.argv[16]: number of stages
    print "number of stages: " + sys.argv[16]
    # sys.argv[17]: keep or not the AR when resizing to crop_size
    print "keep AR?: " + sys.argv[17]
    # sys.argv[18]: number of padded pixels in output image
    print "Padded pixels: " + sys.argv[18]
    # sys.argv[19]: type viewpoint estimation
    print "type viewpoint estimation: " + sys.argv[19]
    # sys.argv[20]: if only azimuth viewpoint
    print "if only azimuth viewpoint: " + sys.argv[20]
    # sys.argv[21]: only VP information or also with Parts
    print "viewpoint bin size: " + sys.argv[21]

    ### Change here for different dataset
    task_name = sys.argv[1]
    directory = sys.argv[2] + '/' + task_name
    dataFolder = sys.argv[3]
    testFolder = sys.argv[4] # dataFolder + '_val'
    dataSynFolder = sys.argv[5]
    dataShapeNetFolder = sys.argv[6]
    patchSize = int(sys.argv[7])
    batch_size_train = int(sys.argv[8])
    batch_size_test = int(sys.argv[9])
    num_iter_cnn = int(sys.argv[10])
    base_lr = float(sys.argv[11])
    step_lr = int(sys.argv[12])
    numClasses = int(sys.argv[13])
    numParts = int(sys.argv[14]) + 1 # adding bg
    sumParts = int(sys.argv[15]) + numClasses # adding bg
    numStages = int(sys.argv[16])
    keepAR = bool(int(sys.argv[17]))
    pad = int(sys.argv[18])
    typeVP = sys.argv[19]
    onlyAZ = bool(int(sys.argv[20]))
    binSize = float(sys.argv[21])

    if not os.path.exists(directory):
        os.makedirs(directory)

    trans_param_train = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize, scale_prob=1,
                             scale_min=0.4, scale_max=1.0, max_rotate_degree=45, center_perterb_max=0.9,
                             do_clahe=False, num_parts=numParts-1, np_in_lmdb=numParts-1, flip_prob=0.5,
                             is_rigid=True, keep_ar=keepAR, pad=pad,
                             type_vp = typeVP, only_azimuth=onlyAZ, size_bin_vp=binSize)
    trans_param_test = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize, scale_prob=1,
                            scale_min=0.4, scale_max=1.0, max_rotate_degree=0, center_perterb_max=1.0,
                            num_parts=numParts - 1, np_in_lmdb=numParts - 1, flip_prob=0, is_rigid=True, keep_ar=keepAR,
                            pad=pad, type_vp=typeVP, only_azimuth=onlyAZ, size_bin_vp=binSize)
    trans_param_shapenet = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize, scale_prob=0,
                                scale_min=0.0, scale_max=1.0, max_rotate_degree=0, center_perterb_max=1.0,
                                num_parts=numParts - 1, np_in_lmdb=numParts - 1, flip_prob=0.5, is_rigid=True,
                                keep_ar=keepAR, pad=0, type_vp=typeVP, only_azimuth=onlyAZ, size_bin_vp=binSize)

    # Stage 1 (from VGG)
    layername = ['C', 'R', 'C', 'R', 'P',  # conv1
                 'C', 'R', 'C', 'R', 'P',  # conv2
                 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R', 'P',  # conv3
                 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'L'] # conv4 (7) + conv5 (2)
    outCH =  [64, 0, 64, 0, 0,
              128, 0, 128, 0, 0,
              256, 0, 256, 0, 256, 0, 256, 0, 0,
              512, 0, 512, 0, 256, 0, 256, 0, 256, 0, 256, 0, 128, 0, 512, 0, sumParts, 0]
    kernel = [3, 0, 3, 0, 2,
              3, 0, 3, 0, 2,
              3, 0, 3, 0, 3, 0, 3, 0, 2,
              3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 1, 0, 1, 0]
    stride = [1, 0, 1, 0, 2,
              1, 0, 1, 0, 2,
              1, 0, 1, 0, 1, 0, 1, 0, 2,
              1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    # Other states (>2)
    if numStages >= 2:
        for s in range(1, numStages):
            layername   += ['@'] + 6 * ['C', 'R'] + ['C', 'L']
            outCH       += [0] + 5 * [128, 0] + [128, 0, sumParts, 0]
            kernel      += [0] + 5 * [7, 0] + [1, 0, 1, 0]
            stride      += [0] + 5 * [3, 0] + [0, 0, 0, 0]

    if binSize > 0:
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
            layername += ['G']
            kernel += [0]
            stride += [0]
            if onlyAZ:
                outCH += [numClasses*2]
            else:
                outCH += [numClasses*6]

    label_name = ['label_1st_lower', 'label_lower']
    writePrototxts(dataFolder, testFolder, dataSynFolder, dataShapeNetFolder, directory, batch_size_train,
                   batch_size_test, num_iter_cnn, base_lr, step_lr, layername, kernel, stride, outCH,
                   trans_param_train, trans_param_test, trans_param_shapenet, task_name, label_name, patchSize,
                   numClasses, numParts, numStages, typeVP, onlyAZ, binSize)