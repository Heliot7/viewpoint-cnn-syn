# py pose/CPM/genLMDB_vp.py Z:\PhD\Data\SynShapeNet\Pascal3D\json_0\all_annotations.json Z:\PhD\Data\SynShapeNet\Pascal3D\lmdb_0\all

import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import caffe
import os.path
import struct
import sys

def writeLMDB(json_path, lmdb_path, validation):

	data = []
	with open(json_path) as data_file:
		data_this = json.load(data_file)
		data_this = data_this['root']
		data = data + data_this
	numSample = len(data)
	print numSample
	# maxSamples = min(numSample, 1200000)
	maxSamples = min(numSample, 100000)

	map_size_samples = maxSamples*1024*1024*2.0 # 256Kb on average / sample # 2 # (2Mb per sample)
    # 0.22 PASCAL3D 0.4 ObjectNet3D
	# 2.0 Freiburg
	# 0.4 EPFL
	# default map_size int(1e10)
	env = lmdb.open(lmdb_path, map_size=map_size_samples)
	txn = env.begin(write=True)

	random_order = np.random.permutation(numSample).tolist()

	isValidationArray = [data[i]['isValidation'] for i in range(numSample)];
	if(validation == 1):
		totalWriteCount = isValidationArray.count(0.0);
	else:
		totalWriteCount = len(data)
	print 'going to write %d images..' % totalWriteCount;
	writeCount = 0

	for count in range(maxSamples):
		idx = random_order[count]
		if (data[idx]['isValidation'] != 0 and validation == 1):
			print '%d/%d skipped' % (count,idx)
			continue
		#print idx

		img = cv2.imread(data[idx]['img_paths'])
		height = img.shape[0]
		width = img.shape[1]
		if height < 16 or width < 16:
			print '%d/%d skipped (too small < 16 [%d,%d])' % (count, idx, width, height)
			continue
		#if(width < 64):
		#	img = cv2.copyMakeBorder(img,0,0,0,64-width,cv2.BORDER_CONSTANT,value=(128,128,128))
		#		print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
		#	cv2.imwrite('padded_img.jpg', img)
		#	width = 64
			# no modify on width, because we want to keep information
		meta_data = np.zeros(shape=(height,width,1), dtype=np.uint8)
		#print type(img), img.shape
		#print type(meta_data), meta_data.shape
		clidx = 0 # current line index
		# dataset name (string)
		for i in range(len(data[idx]['dataset'])):
			meta_data[clidx][i] = ord(data[idx]['dataset'][i])
		clidx = clidx + 1
		# image height, image width
		height_binary = float2bytes(data[idx]['img_height'])
		for i in range(len(height_binary)):
			meta_data[clidx][i] = ord(height_binary[i])
		width_binary = float2bytes(data[idx]['img_width'])
		for i in range(len(width_binary)):
			meta_data[clidx][4+i] = ord(width_binary[i])
		clidx = clidx + 1
		# (a) objId(uint8), isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
		meta_data[clidx][0] = data[idx]['objId']
		meta_data[clidx][1] = data[idx]['isValidation']
		annolist_index_binary = float2bytes(data[idx]['annolist_index'])
		# print "annolist_index: " + str(data[idx]['annolist_index'])
		for i in range(len(annolist_index_binary)): # 1,2,3,4
			meta_data[clidx][2+i] = ord(annolist_index_binary[i])
		count_binary = float2bytes(float(writeCount)) # note it's writecount instead of count!
		for i in range(len(count_binary)):
			meta_data[clidx][6+i] = ord(count_binary[i])
		totalWriteCount_binary = float2bytes(float(totalWriteCount))
		for i in range(len(totalWriteCount_binary)):
			meta_data[clidx][10+i] = ord(totalWriteCount_binary[i])
		clidx = clidx + 1
		# (b) objpos_x (float), objpos_y (float)
		objpos_binary = float2bytes(data[idx]['objpos'])
		for i in range(len(objpos_binary)):
			meta_data[clidx][i] = ord(objpos_binary[i])
		clidx = clidx + 1
		# (b.1) bb
		bb_binary = float2bytes(data[idx]['bb'])
		for i in range(len(bb_binary)):
			meta_data[clidx][i] = ord(bb_binary[i])
		clidx = clidx + 1
		# (vp) vp - viewpoint 3D azimuth, elevation, in-plane rotation
		vp_binary = float2bytes(data[idx]['vp'])
		for i in range(len(vp_binary)):
			meta_data[clidx][i] = ord(vp_binary[i])
		clidx = clidx + 1
		# print meta_data[0:12,0:48]
		# total 7+4*nop lines
		img4ch = np.concatenate((img, meta_data), axis=2)
		img4ch = np.transpose(img4ch, (2, 0, 1))
		#print img4ch.shape
		datum = caffe.io.array_to_datum(img4ch, label=0)
		key = '%07d' % writeCount
		txn.put(key, datum.SerializeToString())
		if(writeCount % 1000 == 0):
			txn.commit()
			txn = env.begin(write=True)
		if (count % 1000 == 0):  # One print every 1000 saved samples
			print 'count: %d/ write count: %d/ randomized: %d/ all: %d' % (count,writeCount,idx,totalWriteCount)
		writeCount = writeCount + 1

	txn.commit()
	env.close()

def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	return struct.pack('%sf' % len(floats), *floats)

if __name__ == "__main__":

	# sys.argv[1]: json file
	print sys.argv[1]
    # sys.argv[2]: lmdb storage folder
	print sys.argv[2]
	isValidation = 0 # 0: val in json, 1: no val in json
	writeLMDB(sys.argv[1], sys.argv[2], isValidation)
