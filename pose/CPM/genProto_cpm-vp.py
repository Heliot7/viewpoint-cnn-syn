import sys
import os
import math
import argparse
import json
import caffe
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayers(data_source, data_test, batch_size, layername, kernel, stride, outCH, label_name, transform_param_in,
              patchSize, numParts, numStages, onlyAZ, binSize, deploy=False):

    n = caffe.NetSpec()
    assert len(layername) == len(kernel)
    assert len(layername) == len(stride)
    assert len(layername) == len(outCH)

    if deploy == False:
        nr = caffe.NetSpec()
        nr.data, nr.tops['label_kp'], nr.tops['label_kp_pos'], nr.tops['label_vp'] = \
            L.PoseData(cpmdata_param=dict(backend=1, source=data_source, batch_size=batch_size),
                       transform_param=transform_param_in, include=dict(phase=caffe.TRAIN), ntop=4)
        train_data = str(nr.to_proto())
        # Test data
        n.data, n.tops['label_kp'], n.tops['label_kp_pos'], n.tops['label_vp'] = \
            L.PoseData(name='data', cpmdata_param=dict(backend=1, source=data_test, batch_size=batch_size),
                       transform_param=transform_param_in, include=dict(phase=caffe.TEST), ntop=4)
        n.tops[label_name[1]], n.tops[label_name[0]] = \
            L.Slice(n.tops['label_kp'], slice_param=dict(axis=1, slice_point=numParts), ntop=2)
        n.Silence_kp_pos = L.Silence(n.tops['label_kp_pos'], ntop=0)
    else:
        n.data = L.Input(input_param=dict(shape=dict(dim=[1, 4, patchSize, patchSize])),
                         transform_param=transform_param_in, ntop=1)

    # something special before everything
    n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
    n.pool_center_lower = L.Pooling(n.center_map, kernel_size=9, stride=8, pool=P.Pooling.AVE)

    # just follow arrays..CPCPCPCPCCCC....
    last_layer = 'image'
    stage = 1
    conv_counter = 1
    pool_counter = 1
    drop_counter = 1
    state = 'image' # can be image or fuse
    share_point = 0

    for l in range(0, len(layername)):
        if layername[l] == 'C':
            if state == 'image':
                if outCH[l] == numParts:  # Fine-tuning for last conv layer of each stage with parts = num obj parts
                    conv_name = 'conv%d_stage%d_FT' % (conv_counter, stage)
                else:
                    conv_name = 'conv%d_stage%d' % (conv_counter, stage)
            else:
                if conv_counter ==  1 or outCH[l] == numParts: # Fine-tuning for last conv layer of each stage with parts = num obj parts
                    conv_name = 'Mconv%d_stage%d_FT' % (conv_counter, stage)
                else:
                    conv_name = 'Mconv%d_stage%d' % (conv_counter, stage)
            if stage == 1:
                lr_m = 5
            else:
                lr_m = 1
            n.tops[conv_name] = L.Convolution(n.tops[last_layer], kernel_size=kernel[l],
                                                  num_output=outCH[l], pad=int(math.floor(kernel[l]/2)),
                                                  param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant'))
            last_layer = conv_name
            if layername[l+1] != 'L':
                if(state == 'image'):
                    ReLUname = 'relu%d_stage%d' % (conv_counter, stage)
                    n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
                else:
                    ReLUname = 'Mrelu%d_stage%d' % (conv_counter, stage)
                    n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
                last_layer = ReLUname
            conv_counter += 1
        elif layername[l] == 'P': # Pooling
            n.tops['pool%d_stage%d' % (pool_counter, stage)] = L.Pooling(n.tops[last_layer], kernel_size=kernel[l], stride=stride[l], pool=P.Pooling.MAX)
            last_layer = 'pool%d_stage%d' % (pool_counter, stage)
            pool_counter += 1
        elif layername[l] == 'L':
            # Loss: n.loss layer is only in training and testing nets, but not in deploy net.
            if deploy == False:
                if stage == 1:
                    n.tops['loss_stage%d' % stage] = L.EuclideanLoss(n.tops[last_layer], n.tops[label_name[0]])
                else:
                    n.tops['loss_stage%d' % stage] = L.EuclideanLoss(n.tops[last_layer], n.tops[label_name[1]])

            stage += 1
            last_connect = last_layer
            last_layer = 'image'
            conv_counter = 1
            pool_counter = 1
            drop_counter = 1
            state = 'image'
        elif layername[l] == 'D':
            if deploy == False:
                n.tops['drop%d_stage%d' % (drop_counter, stage)] = L.Dropout(n.tops[last_layer], in_place=True, dropout_param=dict(dropout_ratio=0.5))
                drop_counter += 1
        elif layername[l] == '@':
            n.tops['concat_stage%d' % stage] = L.Concat(n.tops[last_layer], n.tops[last_connect], n.pool_center_lower, concat_param=dict(axis=1))
            conv_counter = 1
            state = 'fuse'
            last_layer = 'concat_stage%d' % stage
        elif layername[l] == '$':
            if not share_point:
                share_point = last_layer
            else:
                last_layer = share_point
        # Add FC layer
        elif layername[l] == 'K' or layername[l] == 'R':
            n.tops['fc1_stage_vp%d' % numStages] = L.InnerProduct(n.tops['Mconv4_stage%d' % numStages],
                                                              param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)],
                                                              inner_product_param=dict(num_output=outCH[l]))
            # + Loss layer (and Accuracy if classification task)
            if deploy == False:
                if layername[l] == 'K':
                    if onlyAZ:
                        n.tops['loss_stage_vp_az%d' % numStages] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp%d' % numStages], n.tops['label_vp'])
                        n.tops['accuracy_vp_az%d' % numStages] = L.Accuracy(n.tops['fc1_stage_vp%d' % numStages],n.tops['label_vp'])
                    else:
                        n.tops['fc1_stage_vp_az%d' % numStages], n.tops['fc1_stage_vp_el%d' % numStages], n.tops['fc1_stage_vp_th%d' % numStages] = \
                            L.Slice(n.tops['fc1_stage_vp%d' % numStages],slice_param=dict(axis=1,slice_point=[360/binSize,540/binSize]),ntop=3)
                        n.tops['label_vp_az'], n.tops['label_vp_el'], n.tops['label_vp_th'] = L.Slice(n.tops['label_vp'], slice_param=dict(axis=1,slice_point=[1,2]),ntop=3)
                        n.tops['loss_stage_vp_az%d' % numStages] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_az%d' % numStages],n.tops['label_vp_az'])
                        n.tops['accuracy_vp_az%d' % numStages] = L.Accuracy(n.tops['fc1_stage_vp_az%d' % numStages],n.tops['label_vp_az'],
                                                                            include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_el%d' % numStages] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_el%d' % numStages],n.tops['label_vp_el'])
                        n.tops['accuracy_vp_el%d' % numStages] = L.Accuracy(n.tops['fc1_stage_vp_el%d' % numStages],n.tops['label_vp_el'],
                                                                            include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_th%d' % numStages] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_th%d' % numStages],n.tops['label_vp_th'])
                        n.tops['accuracy_vp_th%d' % numStages] = L.Accuracy(n.tops['fc1_stage_vp_th%d' % numStages],n.tops['label_vp_th'],
                                                                            include=dict(phase=caffe.TEST))
                elif layername[l] == 'R':
                    n.tops['loss_stage_vp%d' % numStages] = L.SmoothL1Loss(n.tops['fc1_stage_vp%d' % numStages], n.tops['label_vp'], loss_weight=1)


    # final process
    stage -= 1
    if stage == 1:
        n.silence = L.Silence(n.pool_center_lower, ntop=0)

    if deploy == False:
        return train_data + str(n.to_proto())
    else:
        return 'name:"CPM-VP"\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[1:])


def writePrototxts(dataFolder, testFolder, dir, batch_size, num_iter_cnn, layername, kernel, stride, outCH, transform_param_in,
                   base_lr, task_name, proto_folder, weights_folder, label_name, patchSize, numParts, numStages, onlyAZ, binSize):

    # write the net prototxt files out
    with open('%s/%s_train_test.prototxt' % (dir, task_name.lower()), 'w') as f:
        print 'writing %s/%s_train_test.prototxt' % (dir, task_name.lower())
        str_to_write = setLayers(dataFolder, testFolder, batch_size, layername, kernel, stride, outCH, label_name, transform_param_in,
                                 patchSize, numParts, numStages, onlyAZ, binSize, deploy=False)
        f.write(str_to_write)

    with open('%s/%s_deploy.prototxt' % (dir, task_name.lower()), 'w') as f:
        print 'writing %s/%s_deploy.prototxt' % (dir, task_name.lower())
        str_to_write = str(setLayers('', '', 0, layername, kernel, stride, outCH, label_name, transform_param_in,
                                     patchSize, numParts, numStages, onlyAZ, binSize, deploy=True))
        f.write(str_to_write)

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('%s/%s_solver.prototxt' % (dir, task_name.lower()), "w") as f:
        solver_string = getSolverPrototxt(base_lr, task_name, proto_folder, weights_folder, num_iter_cnn)
        print 'writing %s/%s_solver.prototxt' % (dir, task_name.lower())
        f.write('%s' % solver_string)


def getSolverPrototxt(base_lr, task_name, proto_folder, weights_folder, num_iter_cnn):
    string = 'net: "%s/%s_train_test.prototxt"\n\
# The base learning rate, momentum and the weight decay of the network.\n\
# Evaluation of test data\n\
test_iter: 100\n\
test_interval: 1000\n\
base_lr: %f\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
# The learning rate policy\n\
lr_policy: "step"\n\
gamma: 0.333\n\
stepsize: 25000\n\
# Display every 100 iterations\n\
display: 100\n\
# The maximum number of iterations\n\
max_iter: %d\n\
# snapshot intermediate results\n\
snapshot: %d\n\
snapshot_prefix: "%s/%s"\n\
# solver mode: CPU or GPU\n\
solver_mode: GPU\n' % (proto_folder, task_name, base_lr, num_iter_cnn, num_iter_cnn, weights_folder, task_name)
    return string


if __name__ == "__main__":

    # sys.argv[1]: name task (class included)
    print "task: " + sys.argv[1]
    # sys.argv[2]: folder prototxt folder (solver and model together + caffe weights to store)
    print "prototxt folder: " + sys.argv[2]
    # sys.argv[3]: lmdb folder
    print "lmdb folder: " + sys.argv[3]
    # sys.argv[4]: size patch
    print "size patch: [" + sys.argv[4] + "," + sys.argv[4] + "]"
    # sys.argv[5]: size batch
    print "size batch: [" + sys.argv[5] + "," + sys.argv[5] + "]"
    # sys.argv[6]: number of iterations CNN
    print "num iter CNN: " + sys.argv[6]
    # sys.argv[7]: number of parts
    print "number of parts: " + sys.argv[7]
    # sys.argv[8]: number of stages
    print "number of stages: " + sys.argv[8]
    # sys.argv[10]: type viewpoint estimation
    print "type viewpoint estimation: " + sys.argv[9]
    # sys.argv[11]: if only azimuth viewpoint
    print "if only azimuth viewpoint: " + sys.argv[10]
    # sys.argv[12]: only VP information or also with Parts
    print "viewpoint bin size: " + sys.argv[11]
    # sys.argv[13]: keep or not the AR when resizing to crop_size
    print "keep AR?: " + sys.argv[12]
    # sys.argv[14]: number of padded pixels in output image
    print "Padded pixels: " + sys.argv[13]

    ### Change here for different dataset
    task_name = sys.argv[1]
    directory = sys.argv[2] + '/' + task_name
    dataFolder = sys.argv[3]
    testFolder = dataFolder + '_val'
    patchSize = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    num_iter_cnn = int(sys.argv[6])
    numParts = int(sys.argv[7])+1
    numStages = int(sys.argv[8])
    typeVP = sys.argv[9]
    onlyAZ = bool(int(sys.argv[10]))
    binSize = int(sys.argv[11])
    keepAR = bool(int(sys.argv[12]))
    pad = int(sys.argv[13])
    ###

    weights_folder = '%s' % directory # the place you want to store your caffemodel
    base_lr = 8e-5
    transform_param = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize,
                           target_dist=1.171, scale_prob=0, scale_min=1.0, scale_max=1.0,
                           max_rotate_degree=0, center_perterb_max=0.85, do_clahe=False,
                           num_parts=numParts-1, np_in_lmdb=numParts-1, flip_prob=0.5, is_rigid=True,
                           type_vp=typeVP, only_azimuth=onlyAZ, keep_ar=keepAR, pad=pad, size_bin_vp=binSize)
    nCP = 3
    CH = 128
    if not os.path.exists(directory):
        os.makedirs(directory)

    layername = ['C', 'P'] * nCP + ['C','C','D','C','D','C'] + ['L'] # first-stage
    kernel =    [ 9,  3 ] * nCP + [ 5 , 9 , 0 , 1 , 0 , 1 ] + [0] # first-stage
    outCH =     [128, 128] * nCP + [ 32,512, 0 ,512, 0 , numParts] + [0] # first-stage
    stride =    [ 1 ,  2 ] * nCP + [ 1 , 1 , 0 , 1 , 0 , 1 ] + [0] # first-stage

    if numStages >= 2:
        layername += ['C', 'P'] * nCP + ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['L']
        outCH +=     [128, 128] * nCP + [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,numParts] + [ 0 ]
        kernel +=    [ 9,   3 ] * nCP + [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11, 1,   1] + [ 0 ]
        stride +=    [ 1 ,  2 ] * nCP + [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ]

        for s in range(3, numStages+1):
            layername += ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['L']
            outCH +=     [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,numParts] + [ 0 ]
            kernel +=    [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11,  1, 1 ] + [ 0 ]
            stride +=    [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ]

        if typeVP == "class":
            layername += ['K']
            kernel += [0]
            stride += [0]
            if onlyAZ:
                outCH += [360/binSize]

            else:
                outCH += [900/binSize]
        elif typeVP == "reg":
            layername += ['R']
            kernel += [0]
            stride += [0]
            if onlyAZ:
                outCH += [2]
            else:
                outCH += [6]

    label_name = ['label_1st_lower', 'label_lower']
    writePrototxts(dataFolder, testFolder, directory, batch_size, num_iter_cnn, layername, kernel, stride, outCH, transform_param,
                   base_lr, task_name, directory, weights_folder, label_name, patchSize, numParts, numStages, onlyAZ, binSize)