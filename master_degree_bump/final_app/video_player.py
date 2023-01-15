from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QFileDialog, QLabel, QPushButton, QSlider, QStyle, QWidget, 
                             QGridLayout)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton
import sys

from PyQt5.QtWidgets import  QWidget, QLabel
from PyQt5.QtCore import Qt

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
        self.message = QLabel()
        self.message.setFixedSize(640, 20)
        
        self.system_info = QLabel()
        self.system_info.setFixedSize(640, 20)

        #   set a buttom responsible for opening the video file that will be played
        openButton = QPushButton("Open Video")   
        openButton.setToolTip("Open Video File")
        openButton.setStatusTip("Open Video File")
        openButton.clicked.connect(self.openFile)
        
        save_exit_button = QPushButton('Save reaction file')
        save_exit_button.setToolTip("Save reaction file in a CSV format")
        save_exit_button.setStatusTip('Save file in a CSV file')
        save_exit_button.setFixedSize(1280, 20)
        save_exit_button.clicked.connect(self.save_file)

        #   set layout that will contain all required elements and set a grid where it all will appear
        layout = QGridLayout()
    
        layout.addWidget(videoWidget, 0, 0)
        layout.addWidget(self.webcam_stream, 0, 1)
        layout.addWidget(self.positionSlider, 1, 0)
        layout.addWidget(self.playButton, 1, 1)
        layout.addWidget(openButton, 2, 0)
        layout.addWidget(self.message, 2, 1)
        layout.addWidget(save_exit_button, 3, 0)
        layout.addWidget(self.system_info, 3, 1)
        
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
            self.webcam_stream.set_msg_output(self.message)
            self.webcam_stream.video_thread.start()
            self.webcam_stream.video_thread.PAUSE = True
            self.system_info.setText("File opened, press play and start reacting")
 
    def exitCall(self):
        """close program entirely in case of closing the main window
        """
        sys.exit(app.exec_())
 
    def play(self):
        """handler of the play button event, run if positive and stop if negative
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.webcam_stream.video_thread.PAUSE = True
            self.system_info.setText('Video and reaction are paused')
        else:
            self.mediaPlayer.play()
            self.webcam_stream.video_thread.PAUSE = False
            self.system_info.setText('Resuming reaction and video')
 
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
        self.system_info.setText("Error: " + self.mediaPlayer.errorString())
        
    def save_file(self):
        self.webcam_stream.video_thread.exit()
        self.system_info.setText("Reaction saved, you can close the app")
        