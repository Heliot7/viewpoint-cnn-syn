import sys
import os
import math
import argparse
import json
import caffe
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayers(real_data, real_test, syn_data, batch_size, layername, kernel, stride, outCH, transform_param_in,
              patchSize, typeData, onlyAZ, binSize, deploy=False):

    n = caffe.NetSpec()
    assert len(layername) == len(kernel)
    assert len(layername) == len(stride)
    assert len(layername) == len(outCH)

    # produce data definition for deploy net
    if deploy == False:
        train_data = ''
        nr = caffe.NetSpec()
        name_vp = 'label_vp'
        if typeData == 'real' or typeData == 'joint':
            name_real = 'data'
            div_batch = 1
            if typeData == 'joint':
                name_real = 'data_real'
                name_vp = 'vp_real'
                div_batch = 2
            nr.tops[name_real], nr.tops['label_kp'], nr.tops['label_kp_pos'], nr.tops[name_vp] = \
                L.PoseData(cpmdata_param=dict(backend=1, source=real_data, batch_size=int(batch_size/div_batch)),
                           transform_param=transform_param_in, include=dict(phase=caffe.TRAIN), ntop=4)
            if typeData != 'joint':
                train_data = str(nr.to_proto())
        if typeData == 'shapenet' or typeData == 'joint':
            transform_param_shapenet = transform_param_in.copy()
            transform_param_shapenet['scale_prob'] = 0
            transform_param_shapenet['center_perterb_max'] = 1.0
            transform_param_shapenet['pad'] = 0
            name_syn = 'data'

            if typeData == 'joint':
                name_syn = 'data_syn'
                name_vp = 'vp_syn'
            nr.tops[name_syn], nr.tops[name_vp] = \
                L.ViewpointData(cpmdata_param=dict(backend=1, source=syn_data, batch_size=int(batch_size/div_batch)),
                                transform_param=transform_param_shapenet, include=dict(phase=caffe.TRAIN), ntop=2)
            if typeData == 'joint':
                nr.data = L.Concat(nr.data_real, nr.data_syn, concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
                nr.label_vp = L.Concat(nr.vp_real, nr.vp_syn, concat_param=dict(concat_dim=0), include=dict(phase=caffe.TRAIN))
            train_data = str(nr.to_proto())
        n.data, n.tops['label_kp'], n.tops['label_kp_pos'], n.tops['label_vp'] = \
            L.PoseData(name='data', cpmdata_param=dict(backend=1, source=real_test, batch_size=batch_size),
                       transform_param=transform_param_in, include=dict(phase=caffe.TEST), ntop=4)
        if typeData == 'real' or typeData == 'joint':
            n.Silence_kp = L.Silence(n.tops['label_kp'], ntop=0)
            n.Silence_kp_pos = L.Silence(n.tops['label_kp_pos'], ntop=0)
        else:
            n.Silence_kp = L.Silence(n.tops['label_kp'], include=dict(phase=caffe.TEST), ntop=0)
            n.Silence_kp_pos = L.Silence(n.tops['label_kp_pos'], include=dict(phase=caffe.TEST), ntop=0)

    else:
        n.data = L.Input(input_param=dict(shape=dict(dim=[1, 3, patchSize, patchSize])), transform_param=transform_param_in, ntop=1)

    # just follow arrays..CPCPCPCPCCCC....
    last_layer = 'data'
    layer_counter = 1
    conv_counter = 0

    for l in range(0, len(layername)):
        if layername[l] == 'C':
            conv_counter += 1
            conv_name = 'conv%d_%d' % (layer_counter, conv_counter)
            n.tops[conv_name] = L.Convolution(n.tops[last_layer], kernel_size=kernel[l], num_output=outCH[l], pad=int(math.floor(kernel[l]/2)))
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
            n.tops[fc_name] = L.InnerProduct(n.tops[last_layer], inner_product_param=dict(num_output=outCH[l]))
            last_layer = fc_name
        # Add FC layer
        elif layername[l] == 'K' or layername[l] == 'R':
            n.tops['fc1_stage_vp1'] = L.InnerProduct(n.tops['fc7'], param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                     inner_product_param=dict(num_output=outCH[l]))
            # + Loss layer (and Accuracy if classification task)
            if deploy == False:
                if layername[l] == 'K':
                    if onlyAZ:
                        n.tops['loss_stage_vp_az1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp1'], n.tops['label_vp'])
                        n.tops['accuracy_vp_az1'] = L.Accuracy(n.tops['fc1_stage_vp1'],n.tops['label_vp'])
                    else:
                        n.tops['fc1_stage_vp_az1'], n.tops['fc1_stage_vp_el1'], n.tops['fc1_stage_vp_th1'] = \
                            L.Slice(n.tops['fc1_stage_vp1'],slice_param=dict(axis=1,slice_point=[360/binSize,540/binSize]),ntop=3)
                        n.tops['label_vp_az'], n.tops['label_vp_el'], n.tops['label_vp_th'] = L.Slice(n.tops['label_vp'], slice_param=dict(axis=1,slice_point=[1,2]),ntop=3)
                        n.tops['loss_stage_vp_az1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_az1'],n.tops['label_vp_az'])
                        n.tops['accuracy_vp_az1'] = L.Accuracy(n.tops['fc1_stage_vp_az1'],n.tops['label_vp_az'],include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_el1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_el1'],n.tops['label_vp_el'])
                        n.tops['accuracy_vp_el1'] = L.Accuracy(n.tops['fc1_stage_vp_el1'],n.tops['label_vp_el'],include=dict(phase=caffe.TEST))
                        n.tops['loss_stage_vp_th1'] = L.SoftmaxWithLoss(n.tops['fc1_stage_vp_th1'],n.tops['label_vp_th'])
                        n.tops['accuracy_vp_th1'] = L.Accuracy(n.tops['fc1_stage_vp_th1'],n.tops['label_vp_th'],include=dict(phase=caffe.TEST))
                elif layername[l] == 'R':
                    n.tops['loss_stage_vp1'] = L.SmoothL1Loss(n.tops['fc1_stage_vp1'], n.tops['label_vp'], loss_weight=1)

    if deploy == False:
        return train_data + str(n.to_proto())
    else:
        return 'name: "VP"\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[1:])


def writePrototxts(dataFolder, testFolder, synDataFolder, dir, batch_size, num_iter_cnn, layername,
                   kernel, stride, outCH, transform_param_in, base_lr, task_name, proto_folder, weights_folder,
                   patchSize, typeData, onlyAZ, binSize):
    # write the net prototxt files out
    with open('%s/%s_train_test.prototxt' % (dir, task_name.lower()), 'w') as f:
        print 'writing %s/%s_train_test.prototxt' % (dir, task_name.lower())
        str_to_write = setLayers(dataFolder, testFolder, synDataFolder, batch_size, layername, kernel, stride, outCH,
                                 transform_param_in, patchSize, typeData, onlyAZ, binSize, deploy=False)
        f.write(str_to_write)

    with open('%s/%s_deploy.prototxt' % (dir, task_name.lower()), 'w') as f:
         print 'writing %s/%s_deploy.prototxt' % (dir,task_name.lower())
         str_to_write = str(setLayers('', '', '', 0, layername, kernel, stride, outCH, transform_param_in,
                                      patchSize, typeData, onlyAZ, binSize, deploy=True))
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
gamma: 0.1\n\
stepsize: 20000\n\
# Display every 100 iterations\n\
display: 1000\n\
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
    # sys.argv[3]: lmdb folder (real)
    print "lmdb real folder: " + sys.argv[3]
    # sys.argv[4]: lmdb folder (syn)
    print "lmdb syn folder: " + sys.argv[4]
    # sys.argv[5]: size patch
    print "size patch: [" + sys.argv[5] + "," + sys.argv[5] + "]"
    # sys.argv[6]: size batch
    print "size batch: " + sys.argv[6]
    # sys.argv[7]: number of iterations CNN
    print "num iter CNN: " + sys.argv[7]
    # sys.argv[8]: typeData - real, syn or joint
    print "typeData: " + sys.argv[8]
    # sys.argv[9]: regression or classification
    print "type viewpoint estimation: " + sys.argv[9]
    # sys.argv[10]: if only azimuth viewpoint
    print "if only azimuth viewpoint: " + sys.argv[10]
    # sys.argv[11]: bin size of viewpoint discretisation
    print "viewpoint bin size: " + sys.argv[11]
    # sys.argv[12]: keep or not the AR when resizing to crop_size
    print "keep AR?: " + sys.argv[12]
    # sys.argv[13]: number of padded pixels in output image
    print "Padded pixels: " + sys.argv[13]

    ### Change here for different dataset
    task_name = sys.argv[1]
    directory = sys.argv[2] + '/' + task_name
    dataFolder = sys.argv[3]
    testFolder = dataFolder + '_val'
    synDataFolder = sys.argv[4]
    patchSize = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    num_iter_cnn = int(sys.argv[7])
    typeData = sys.argv[8]
    typeVP = sys.argv[9]
    onlyAZ = bool(int(sys.argv[10]))
    binSize = int(sys.argv[11])
    keepAR = bool(int(sys.argv[12]))
    pad = int(sys.argv[13])
    ###

    weights_folder = '%s' % directory # the place you want to store your caffemodel
    transform_param = dict(stride=8, crop_size_x=patchSize, crop_size_y=patchSize,
                           target_dist=1.171, scale_prob=0, scale_min=1.0, scale_max=1.0,
                           max_rotate_degree=0, center_perterb_max=0.8, do_clahe=False, put_gaussian=False,
                           flip_prob=0.5, is_rigid=True, type_vp=typeVP, only_azimuth=onlyAZ, keep_ar=keepAR, pad=pad,
                           size_bin_vp=binSize)

    if not os.path.exists(directory):
        os.makedirs(directory)

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

    base_lr = 0.001
    if typeVP == "class":
        base_lr = 0.001
        layername += ['K']
        kernel += [0]
        stride += [0]
        if onlyAZ:
            outCH += [360/binSize]

        else:
            outCH += [900/binSize]
    elif typeVP == "reg":
        base_lr = 0.0001
        layername += ['R']
        kernel += [0]
        stride += [0]
        if onlyAZ:
            outCH += [2]
        else:
            outCH += [6]

    writePrototxts(dataFolder, testFolder, synDataFolder, directory, batch_size, num_iter_cnn, layername, kernel,
                   stride, outCH, transform_param, base_lr, task_name, directory, weights_folder, patchSize,
                   typeData, onlyAZ, binSize)