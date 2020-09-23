# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow import nest
import tensorflow_datasets as tfds
import glob


from stacked_capsule_autoencoders.capsules.data import tfrecords as _tfrecords


def create(which,
           batch_size,
           subset=None,
           n_replicas=1,
           transforms=None,
           **kwargs):
  """Creates data loaders according to the dataset name `which`."""

  func = globals().get('_create_{}'.format(which), None)
  if func is None:
    raise ValueError('Dataset "{}" not supported. Only {} are'
                     ' supported.'.format(which, SUPPORTED_DATSETS))

  dataset = func(subset, batch_size, **kwargs)

  if transforms is not None:
    if not isinstance(transforms, dict):
      transforms = {'image': transforms}

    for k, v in transforms.items():
      transforms[k] = snt.Sequential(nest.flatten(v))

  if transforms is not None or n_replicas > 1:

    def map_func(data):
      """Replicates data if necessary."""
      data = dict(data)

      if n_replicas > 1:
        tile_by_batch = snt.TileByDim([0], [n_replicas])
        data = {k: tile_by_batch(v) for k, v in data.items()}

      if transforms is not None:
        img = data['image']

        for k, transform in transforms.items():
          data[k] = transform(img)

      return data

    dataset = dataset.map(map_func)

  iter_data = dataset.make_one_shot_iterator()
  input_batch = iter_data.get_next()
  for _, v in input_batch.items():
    v.set_shape([batch_size * n_replicas] + v.shape[1:].as_list())

  return input_batch


def _create_mnist(subset, batch_size, **kwargs):
  return tfds.load(
      name='mnist', split=subset, **kwargs).repeat().batch(batch_size)

def _create_dataset256(subset, batch_size, **kwargs):
    img_list = []
    images_path = sorted(glob.glob('/home/gpu/qianyu/dataset_256/{}/images/*.png'.format(subset)))
    for img in images_path:
      # image_string = tf.read_file(img)
      img_1 = tf.io.read_file(img)
      n = tf.image.decode_png(img_1)
      n = tf.image.rgb_to_grayscale(n, name=None)
      # n = tf.image.resize(n,size = [300,300])
      n = tf.cast(n,tf.float32)
      img_list.append(n)
    return tf.data.Dataset.from_tensor_slices({'image':img_list,'label':np.random.randint(10, size=len(img_list), dtype=np.int64)}).repeat().batch(batch_size)

def _create_ucmerced(subset, batch_size, **kwargs):
    img_list = []
    kind_list = []
    images_path = sorted(glob.glob('/home/gpu/qianyu/UCMerced_LandUse/{}/*.png'.format(subset)))
    for img in images_path:
      # image_string = tf.read_file(img)
      img_1 = tf.io.read_file(img)
      n = tf.image.decode_png(img_1)
      n = tf.image.rgb_to_grayscale(n, name=None)
      n = tf.image.resize(n, size=[256,256])
      n = tf.cast(n,tf.float32)
      img_list.append(n)
      kind = img.split('/')[-1][0]
      if kind == 'a':
        kind = 0
      elif kind == 'b':
        kind = 1
      elif kind == 'f':
        kind = 2
      elif kind == 'h':
        kind = 3
      else:
        kind = 4
      kind_list.append(kind)
    sh = np.arange(len(img_list))
    np.random.shuffle(sh)
    img_list = [img_list[x] for x in sh]
    kind_list = np.array([kind_list[x] for x in sh], dtype=np.int64)
    return tf.data.Dataset.from_tensor_slices({'image':img_list,'label':kind_list}).repeat().batch(batch_size)


SUPPORTED_DATSETS = set(
    k.split('_', 2)[-1] for k in globals().keys() if k.startswith('_create'))
