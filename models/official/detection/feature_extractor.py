# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
# pylint: disable=line-too-long
r"""A stand-alone binary to run model inference and visualize results.

It currently only supports model of type `retinanet` and `mask_rcnn`. It only
supports running on CPU/GPU with batch size 1.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from configs import factory as config_factory
from dataloader import mode_keys
from modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict
       


class FeatureExtractor():
    def __init__(self, checkpoint_path: str, image_size: int, config_file: str) -> None:

        params = config_factory.config_generator('segmentation')
        params = params_dict.override_params_dict(
            params, config_file, is_strict=True)
        params.override({
            'architecture': {
                'use_bfloat16': False,  # The inference runs on CPU/GPU.
                            },
                        },
                        is_strict=True)
        params.validate()
        params.lock()

        model = model_factory.model_generator(params)        

        _graph = tf.Graph()
        with _graph.as_default():
            self.input = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32)
            # model inference
            self.outputs = model.build_outputs(
            self.input, {}, mode=mode_keys.PREDICT)
            saver = tf.train.Saver()

        # allow growth to use it simultaneously with other methods GPU methods        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=_graph, config=config)
        saver.restore(self.sess, checkpoint_path)

    def __call__(self, batch: np.ndarray) -> dict:
        predictions_np = self.sess.run(self.outputs, feed_dict= {self.input: batch})
        return predictions_np

