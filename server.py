#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

import ConfigParser as configparser
import tornado.web
import tornado.options
import numpy as np

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    return parser.parse_args()

def read_config():
    config = configparser.ConfigParser()
    config.readfp(open(os.path.join(os.path.dirname(__file__), 'config.ini')))
    return config

class Application(tornado.web.Application):
    def __init__(self, config, **kwargs):
        kwargs['handlers'] = [
            (r'/analyse', AnalyseHandler)
        ]
        super(Application, self).__init__(**kwargs)
        merge_cfg_from_file(config.get('DETECTRON', 'CONFIG'))
        cfg.TEST.WEIGHTS = config.get('DETECTRON', 'WEIGHTS')
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg()
        self.model = infer_engine.initialize_model_from_cfg()
#        dummy_coco_dataset = dummy_datasets.get_coco_dataset()      

class AnalyseHandler(tornado.web.RequestHandler):
    def post(self):
        if 'image' not in self.request.files:
            raise tornado.web.HTTPError(400, "missing field 'image'")
        image = self.request.files['image'][0]
        if not image['content_type'].startswith('image/'):
            raise tornado.web.HTTPError(400, "wrong content-type '{}'. Expecting 'image/*'".format(image['content_type']))
        
        nparr = np.fromstring(image['body'], np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        timers = defaultdict(Timer)
        try:
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    self.application.model, im, None, timers=timers
                )
        except Exception as e:
            print(e)
            raise tornado.web.HTTPError(500)
        print(cls_boxes)
        print(cls_segms)
        print(cls_keyps)
        print(timers)
        return
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )


def main(args, config):
    logger = logging.getLogger(__name__)

    tornado.options.parse_command_line()
    application = Application(config, debug=True, autoreload=True)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(config.getint('SERVICE', 'PORT'))
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    config = read_config()
    args = parse_args()
    main(args, config)
