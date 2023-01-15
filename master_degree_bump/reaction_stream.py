from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QGridLayout,
        QMessageBox)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
import sys

import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

#   initially it was planned to work with the PyQt6, but there are problems with running video
# because some components are still in development and therefore here will be used PyQt5

class Thread(QThread):
    """To evade performance problems it is required to set a thread that will run in the background
    with video feed taken from the webcam
    """
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
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                
                #   after performed adaptations send image as a signal to the updatable QT frame/section
                self.changePixmap.emit(p)


class App(QWidget):
    """general application object to either launch it separately, or to make it integratable as
    submodule to application (second scenario used in this case)
    """
    def __init__(self):
        """create it as a QT widget element adding name, offset, size and initializing necessary
        components
        """
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 720
        self.initUI()

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
        
        self.video_thread = Thread(self)
        self.video_thread.changePixmap.connect(self.setImage)
        self.show()

 
class VideoPlayer(QMainWindow):
    """Video player that is based on the PyQt5 components and functionalities. IMPORTANT:
    in case if there will be any problem with running the video it is required to install
    codecs to the system that will be able to run specified files (the problem is that
    code calls for the system codecs to play videos and without them nothing will work).
    RECOMMENDED: install K-lite basic codecs
    """
    def __init__(self, webcam_stream_widget: QWidget):
        super().__init__()
        self.setWindowTitle("PyQt5 Video Player") 
 
        #   considering that videos can differ by their size and that most of the videos have
        # 16:9 ratio, therefore is taken HD quality video as a base case
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        videoWidget.setFixedSize(1280, 720)
 
        #   set play button with negative (not playing) state in the beginning, set icon for it
        # connect this button with play functionality of the framework
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
 
        #   set a slider covering progress of playing video, connect slider to the slot that
        # will connect it to the video playing progress
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setFixedSize(1280, 20)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        
        #   initialize video stream entity
        self.webcam_stream = webcam_stream_widget
 
        #   make element where errors will be written
        self.error = QLabel()
        self.error.setFixedSize(640, 20)

        #   set layout that will contain all required elements and set a grid where it all will appear
        layout = QGridLayout()
    
        layout.addWidget(videoWidget, 0, 0)
        layout.addWidget(self.webcam_stream, 0, 1)
        layout.addWidget(self.positionSlider, 1, 0)
        layout.addWidget(self.playButton, 1, 1)
        layout.addWidget(openButton, 2, 0)
        layout.addWidget(self.error, 2, 1)
        layout.addWidget(save_exit_button, 3, 0)
        
        #   set up a central element of the application and insert formed layout
        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)
        
        #   set video player, set handler of changed position, changed play state, duration of the video
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
 
    def openFile(self):
        """Call for the system window to choose a video file that will be opened
        """
        #   open system window to pick a file to watch
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())
        
        #   set media to be watched and change play state only if there is chosen file
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.webcam_stream.video_thread.start()
 
    def exitCall(self):
        """close program entirely in case of closing the main window
        """
        sys.exit(app.exec_())
 
    def play(self):
        """handler of the play button event, run if positive and stop if negative
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
 
    def mediaStateChanged(self, state):
        """handler of the play state. used to change play icon depending on the state
        :param state: state of the player
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
 
    def positionChanged(self, position):
        """handler that will update state of the slider depending on the media progress
        :param position: where player is actually at the moment
        """
        self.positionSlider.setValue(position)
 
    def durationChanged(self, duration):
        """handler that updates duration of the video depending on the video
        :param duration: how long video goes
        """
        self.positionSlider.setRange(0, duration)
 
    def setPosition(self, position):
        """set new play position of the video using the slider component
        :param position: position of the slider element
        """
        self.mediaPlayer.setPosition(position)
 
    def handleError(self):
        """handler of any error that will happen
        """
        self.playButton.setEnabled(False)
        self.error.setText("Error: " + self.mediaPlayer.errorString())
 
#   initialize all elements of the application
app = QApplication(sys.argv)
webcam_stream = App()
videoplayer = VideoPlayer(webcam_stream)
videoplayer.resize(1920, 760)
videoplayer.show()
sys.exit(app.exec_())