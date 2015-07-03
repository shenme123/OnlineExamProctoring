from fpp import FacePP
import cv2
from status_def import *
import Queue
from gaze import get_gaze
import os


def ann_face(img, faces, color=(0, 0, 255)):
    width = img.shape[1]
    height = img.shape[0]
    for face in faces:
        w = face[u'position'][u'width']*width/100
        h = face[u'position'][u'width']*height/100
        center = face[u'position'][u'center']
        cv2.rectangle(img, (int(center[u'x']*width/100 - w/2), int(center[u'y']*height/100 - h/2)),
                      (int(center[u'x']*width/100 + w/2), int(center[u'y']*height/100 + h/2)), color)

def ann_landmark(img, landmark):
    for fea in landmark:
        if fea.find("countour")>=0:
            point = landmark[fea]
            img = cv2.circle(img,(int(point["x"]*img.shape[1]/100),int(point["y"]*img.shape[0]/100)), 2, (255, 0, 0), -1)


def ann_gaze(img, gaze_info, face):
    landmark = face[u'landmark']
    for fea in landmark:
        if fea.find("eye")>=0 or fea.find("nose")>=0 :
            point = landmark[fea]
            img = cv2.circle(img,(int(point["x"]*img.shape[1]/100),int(point["y"]*img.shape[0]/100)), 2, (255, 0, 0), -1)
    circle_l = gaze_info[-2]
    circle_r = gaze_info[-1]
    cv2.circle(img, (circle_l[0], circle_l[1]), circle_l[2], (0,255,0), 1)
    cv2.circle(img, (circle_l[0], circle_l[1]), 2, (0,0,255), 1)
    cv2.circle(img, (circle_r[0], circle_r[1]), circle_r[2], (0,255,0), 1)
    cv2.circle(img, (circle_r[0], circle_r[1]), 2, (0,0,255), 1)

class Proctor:
    def __init__(self):
        self.frames = []
        self.status = {}
        self.reasons = {}
        self.faces_train = []
        self.fpp = FacePP()
        self.trained = False
        self.person = self.fpp.person.create()
        self.info_queue = Queue.Queue()
        for f in os.listdir('frames'):
            os.remove('frames\\'+f)

    def status_name(self, status):
        msg = ''
        if status == S_NORMAL:
            msg += "Normal"
        elif status == S_ABSENT:
            msg += "Absent"
        elif status == S_MULTIPERSON:
            msg += "MultiplePerson"
        elif status == S_TOOFAR:
            msg += "TooFar"
        elif status == S_WRONGPERSON:
            msg += "WrongPerson"
        elif status == S_NONGAZE:
            msg += "NonGazeing"
        elif status == S_TIMEOUT:
            msg += "TimeOut"
        return msg

    def new_frame(self, frame):
        self.frames.append(frame)
        ind = len(self.frames)-1
        (status, reason) = self.check_frame(ind)
        self.status[ind] = status
        self.reasons[ind] = reason
        self.info_queue.put((status, reason))

    def check_frame(self, ind):
        frame = self.frames[ind]
        ann_frame = frame.copy()
        fpp = self.fpp
        ret = (S_NORMAL,{})
        try:
            result = fpp.detect(frame)
            #print result
            if len(result[u'face'])== 0:
                #print 'S_ABSENT'
                ret = (S_ABSENT, {'face_count':0})
            elif len(result[u'face'])> 1:
                #print 'S_MULTIPERSON'
                ann_face(ann_frame, result[u'face'])
                ret = (S_MULTIPERSON, {'face_count':len(result[u'face'])})
            else:
                face = result[u'face'][0]
                ann_face(ann_frame, [face], (0, 255, 0))
                if face[u'position'][u'width']*face[u'position'][u'height']<1000:
                    ann_face(ann_frame,[face])
                    ret = (S_TOOFAR, result)
                else:
                    landmark = fpp.landmark(face[u'face_id'])
                    ann_landmark(ann_frame, landmark)
                    face[u'landmark'] = landmark[u'result'][0][u'landmark']
                    if len(self.faces_train)<5:
                        self.faces_train.append(face)
                    elif len(self.faces_train)==5:
                        self.faces_train.append(face)
                        for f in self.faces_train:
                            fpp.person.add_face(person_id=self.person[u'person_id'], face_id=f[u'face_id'])
                        fpp.train.verify(person_id=self.person[u'person_id'])
                        print 'Trainning person model'
                        self.trained = True
                    else:
                        result = fpp.recognition.verify(person_id=self.person[u'person_id'], face_id=face[u'face_id'])
                        if not result[u'is_same_person']:
                            #print 'S_WRONGPERSON'
                            ann_face(ann_frame,[face])
                            ret = (S_WRONGPERSON, result)
                        else:
                            #print 'Right Person'
                            is_gaze, info = self.check_gaze(face, ind)
                            ann_gaze(ann_frame, info, face)
                            if not is_gaze:
                                #print 'S_NONGAZE'
                                ret = (S_NONGAZE, info)
        except KeyboardInterrupt:
            raise
        except:
            ret = (S_TIMEOUT, {})
           # print sys.exc_info()[0]
            #print('Network problem, Fail test')

        cv2.imwrite('frames\\'+str(ind)+'_'+self.status_name(ret[0])+'.jpg', ann_frame)
        return ret



    def check_gaze(self, face, ind):
        result = get_gaze(self.frames[ind], face)
        print ind, result
        return result[0], result[1:]

    def __del__(self):
        self.fpp.person.delete(person_id=self.person[u'person_id'])