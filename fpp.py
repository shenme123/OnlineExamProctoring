#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path
import cv2
from facepp.facepp import API, File
import random
import string

class FacePP(API):
    def __init__(self):
        if sys.version_info.major != 2:
            sys.exit('Python 2 is required to run this program')

        fdir = None
        if hasattr(sys, "frozen") and \
                sys.frozen in ("windows_exe", "console_exe"):
            fdir = os.path.dirname(os.path.abspath(sys.executable))
            sys.path.append(fdir)
            fdir = os.path.join(fdir, '..')
        else:
            fdir = os.path.dirname(__file__)

        self.fdir = fdir
        with open(os.path.join(fdir, 'facepp','apikey.cfg')) as f:
            exec(f.read())

        srv = locals().get('SERVER')

        API.__init__(self, API_KEY, API_SECRET, srv = srv, timeout=3)
    def temp_name(self):
        return os.path.join(self.fdir, 'temp',(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))) + '.png')

    def detect(self, img):
        fn = self.temp_name()
        cv2.imwrite(fn, img)
        result = self.detection.detect(img = File(fn), attribute=('gender','age','race','smiling', 'glass', 'pose'))
        os.remove(fn)
        return result
    def landmark(self, face_id):
        return self.detection.landmark(face_id=face_id)

if __name__ == '__main__':
    fpp = FacePP()
    f = fpp.detect(cv2.imread('facepp\\testt.png'))
    print f
    print fpp.landmark(f['face'][0]['face_id'])