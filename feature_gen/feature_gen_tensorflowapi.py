

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import h5py
import logging
logging.basicConfig(filename = 'app.log', level = logging.INFO)
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import sys
sys.path.append('/home/alas79923/models/research/object_detection')
from utils import label_map_util
MODEL_NAME = 'faster_rcnn_nas_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def load_train_image():
  train_path = '/tmp2/train2014/'
  valid_path = '/tmp2/val2014/'
  f = open('/tmp2/train_nas_h5/guesswhat.train.new.jsonl','r')
  l = []
  l2 = []
  for line in f:
    jfile = json.loads(line)
    file_name = jfile['image']['file_name']
    l2.append(file_name)
    if 'train' in file_name:
      l.append(train_path+file_name)
    elif 'val' in file_name:
      l.append(valid_path+file_name)
    else:
      print ('file name error', file_name)
  return l,l2


def load_image_into_numpy_array(image):
  if image.mode != 'RGB':
    image = image.convert('RGB')
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

output = h5py.File('/tmp2/train_nas_h5/train_concise.hdf5','w')
feature_list = []
image_paths,image_names = load_train_image()
image_to_idx = {}
idx_to_image = {}
image_to_idx_file = open('/tmp2/train_nas_h5/image_to_idx.json','w')
idx_to_image_file = open('/tmp2/train_nas_h5/idx_to_image.json','w')


index = 0
for image_path in image_paths:
  if image_path not in image_to_idx:
    image_to_idx[image_path] = index
    idx_to_image[index] = image_path
    index += 1
print ('total image number:',len(image_to_idx))
json.dump(image_to_idx,image_to_idx_file)
json.dump(idx_to_image,idx_to_image_file)
batch_size = 18
IMAGE_SIZE = (300, 300)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph,config=config) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    AvgPool = detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')
    # print (len(image_paths))
    one_batch_image = []
    count = 0
    try:
      for i in range(len(idx_to_image)):
        count += 1
        image_path = idx_to_image[i]
        image = Image.open(image_path).resize(IMAGE_SIZE)
        image_np = load_image_into_numpy_array(image)
        one_batch_image.append(image_np)
        if i > 0 and (i+1) % batch_size == 0:
          a = sess.run(AvgPool,feed_dict={image_tensor: np.array(one_batch_image)})
          feature = np.squeeze(np.array(a))
          feature = feature.reshape(-1,50,4032)
          if i == batch_size:
            feature_list = [feature]
          else:
            print (i)
            feature_list.append(feature)
            # feature_list = np.concatenate((feature_list,feature),axis=0)
          one_batch_image = []
        # if i % 1024 == 0 and i > 0:
        #  output.create_dataset(str(count/1024), data=feature_list)
        #  feature_list = []
      feature_list = np.array(feature_list).reshape(-1,50,4032)
      output.create_dataset('0',data=feature_list)
    except IOError as e:
      logging.exception(str(e))
    if len(idx_to_image) % batch_size != 0:
      a = sess.run(AvgPool,feed_dict={image_tensor: np.array(one_batch_image)})
      feature = np.squeeze(np.array(a)).reshape(-1,50,4032)
      output.create_dataset('1', data=feature)
      # np.savetxt('/tmp2/train_nas_feature/'+image_names[i]+'.out',feature)
    # /tmp2/train2014/COCO_train2014_000000291822.jpg
    # /home/alas79923/vqa/faster-rcnn.pytorch/

output.close()
