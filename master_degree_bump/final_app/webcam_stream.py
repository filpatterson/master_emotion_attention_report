from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QWidget)
from PyQt5.QtWidgets import QWidget

import cv2
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import pandas as pd
import time

class Thread(QThread):
    """To evade performance problems it is required to set a thread that will run in the background
    with video feed taken from the webcam
    """
    def __init__(self, mesh_drawer, mesh_collecter, classification_model):
        """initialize web taker thread that will read video stream, draw scan over the face
        and make classification of the model
        """
        super().__init__()
        self.mesh_drawer = mesh_drawer
        self.classification_model = classification_model
        self.mesh_collecter = mesh_collecter
        self.msg_label = None
        self.record_story = []
        self.PAUSE = True
    
    
    #   establishing frame that will be updated as a signal when received new image (frame)
    changePixmap = pyqtSignal(QImage)

    def run(self):
        #   using computer vision module take webcam feed
        cap = cv2.VideoCapture(0)
        
        #   make continous read
        while True:
            ret, frame = cap.read()
            
            #   read image if everything is ok and frame is receivable
            if ret:
                #   take colored image, its shape and find bytes count per image line
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                
                #   original computer vision image can not be processed by the QT and to make it compatible
                # conversion is done
                landmarked_image = self.mesh_drawer.transform(rgbImage)
                if landmarked_image is not None:
                    convertToQtFormat = QImage(landmarked_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                
                if not self.PAUSE:
                    landmarks = self.mesh_collecter.transform(rgbImage)
                    if landmarks is not None:
                        prediction = self.classification_model.predict(landmarks.reshape(1, 468, 3))[0]
                        max_probability_class = np.argmax(prediction)
                        max_probability = max(prediction)
                        # print(f"class = {max_probability_class}; confidence = {max_probability * 100}%")
                        self.msg_label.setText(f"class={max_probability_class}; confidence={max_probability*100}%")
                        self.record_story.append({'time': time.time(),'emotion': max_probability_class, "confidence": max_probability*100})
                
                #   after performed adaptations send image as a signal to the updatable QT frame/section
                self.changePixmap.emit(p)
                
                
    def exit(self):
        df = pd.DataFrame(self.record_story)
        df.to_csv(f"{time.time()}_session.csv", index=False, header=True)


class App(QWidget):
    """general application object to either launch it separately, or to make it integratable as
    submodule to application (second scenario used in this case)
    """
    def __init__(self, mesh_drawer, mesh_collecter, classification_model):
        """create it as a QT widget element adding name, offset, size and initializing necessary
        components
        """
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 720
        
        self.video_thread = Thread(mesh_drawer, mesh_collecter, classification_model)
        self.initUI()
        
    def set_msg_output(self, msg_label: QLabel):
        self.video_thread.msg_label = msg_label

    @pyqtSlot(QImage)
    def setImage(self, image):
        """function to establish signal channel between webcam and frame that will be updated
        """
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(self.width, self.height)
        
        # create a label
        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(self.width, self.height)
        
        self.video_thread.changePixmap.connect(self.setImage)
        self.show()
 