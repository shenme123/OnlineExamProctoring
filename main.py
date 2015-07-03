import cv2
from PyQt4 import QtGui, QtCore, QtWebKit, Qt
from PyQt4.QtGui import *
from proctor import *
from status_def import *
import threading
import subprocess

output_video = 'output'

def QImagefcv(cvimg):

    cvrgb = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    height, width, byte_value = cvrgb.shape
    byte_value = byte_value * width
    return QImage(cvrgb,width, height, byte_value,QImage.Format_RGB888)

class Capture():
    def __init__(self, window):
        self.capturing = False
        self.window = window
        self.c = cv2.VideoCapture(0)
        self.fps = 20
        self.sample_rate = 20
        self.sample_counter = 0

    def start(self):
        print "pressed Start"
        self.capturing = True
        self.sample_counter = self.sample_rate
        self.window.info_label.setText("Started Exam")
        self.proctor = Proctor()
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter(output_video+'.avi',fourcc, self.fps, (640,480))
        #self.c = cv2.VideoCapture('input.mp4')


    def status_msg(self, status, reason):
        msg = ''
        if status == S_NORMAL:
            msg += "Normal"
        elif status == S_ABSENT:
            msg += "Absent: \n no face detected"
        elif status == S_MULTIPERSON:
            msg += "Multiple Person: \n more than one face"
        elif status == S_TOOFAR:
            msg += "Too Far: \n face detected, but too small"
        elif status == S_WRONGPERSON:
            msg += "Wrong Person: \n face mismatch\n"
            msg += "Confidence: " + str(reason[u'confidence'])
        elif status == S_NONGAZE:
            msg += "Non Gazeing: \n looking off the screen\n"
        elif status == S_TIMEOUT:
            msg += "Request Time out."
        return msg

    def capture(self, force=False):
        cap = self.c
        ret, frame = cap.read()
        self.window.img_label.setPixmap(QtGui.QPixmap.fromImage(QImagefcv(frame)))

        self.sample_counter += 1
        if self.capturing:
            if self.proctor.trained:
                self.window.info_label.setText('Face Trained')
            self.out.write(frame)
            if not self.proctor.info_queue.empty():
                status, reason = self.proctor.info_queue.get()
                self.window.status_label.setText(self.status_msg(status, reason))
                if status == S_NORMAL:
                    self.window.status_label.setStyleSheet('color: green')
                else:
                    self.window.status_label.setStyleSheet('color: red')
            if self.sample_counter>=self.sample_rate:
                #print "start testing"
                self.sample_counter = 0
                test = threading.Thread(target=self.proctor.new_frame, args=(frame,))
                test.daemon = True
                test.start()
                #status, reason = self.proctor.new_frame(frame)
                #self.window.status_label.setText(str(status))



    def end(self):
        print "pressed End"
        self.capturing = False
        # cv2.destroyAllWindows()
        self.window.info_label.setText("End Exam")
        #self.proctor.release()
        self.out.release()
        subprocess.Popen('ffmpeg -y -i '+output_video+'.avi '+output_video+'.mp4')
        with open('result.js', 'w') as r:
            r.write("var res_interval = "+str(float(self.sample_rate)/self.fps)+";\n")
            s = [0]*len(self.proctor.status)
            for k in self.proctor.status:
                s[k] = self.proctor.status[k]
            r.write("var res_status = "+repr(s)+";\n")


    def quit(self):
        print "pressed Quit"
        #self.end()
        cap = self.c
        cap.release()
        QtCore.QCoreApplication.quit()


class Window(QWidget):
    def __init__(self):

        QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.img_label = QtGui.QLabel(self)
        self.info_label = QtGui.QLabel(self)
        self.status_label = QtGui.QLabel(self)
        self.info_label.setText("Exam Not Start")

        self.capture = Capture(self)

        self.start_button = QtGui.QPushButton('Start',self)
        self.start_button.clicked.connect(self.capture.start)

        self.end_button = QtGui.QPushButton('Submit',self)
        self.end_button.clicked.connect(self.capture.end)

        self.quit_button = QtGui.QPushButton('Quit',self)
        self.quit_button.clicked.connect(self.capture.quit)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.capture.capture)
        self.timer.start(1000/self.capture.fps)

        rtbox = QtGui.QHBoxLayout()
        rtbox.addWidget(self.img_label)
        rtbox.addWidget(self.status_label)
        rtbox.addWidget(self.info_label)

        rbbox = QtGui.QHBoxLayout()
        rbbox.addWidget(self.start_button)
        rbbox.addWidget(self.end_button)
        rbbox.addWidget(self.quit_button)

        rbox = QtGui.QVBoxLayout()
        ans_page = QtWebKit.QWebView()
        ans_page.load(QtCore.QUrl("sample_choices.html"))

        rbox.addLayout(rtbox)
        rbox.addWidget(ans_page)
        rbox.addLayout(rbbox)


        mainbox = QtGui.QHBoxLayout()
        info_page = QtWebKit.QWebView()
        info_page.load(QtCore.QUrl("sample_question.html"))
        #info_edit.setText(open('sample_question.txt').read())
        mainbox.addWidget(info_page)
        mainbox.addLayout(rbox)

        self.setLayout(mainbox)
        #self.setGeometry(100,100,200,200)
        #self.show()
        self.capture.capture(force=True)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.showFullScreen()


if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
