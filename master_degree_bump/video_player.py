from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
import sys

#   initially it was planned to work with the PyQt6, but there are problems with running video
# because some components are still in development and therefore here will be used PyQt5
 
class VideoPlayer(QMainWindow):
    """Video player that is based on the PyQt5 components and functionalities. IMPORTANT:
    in case if there will be any problem with running the video it is required to install
    codecs to the system that will be able to run specified files (the problem is that
    code calls for the system codecs to play videos and without them nothing will work).
    RECOMMENDED: install K-lite basic codecs
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Video Player") 
 
        #   initialize media player entity where video will be shown, make videowidget
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
 
        #   set play button with negative (not playing) state in the beginning, set icon for it
        # connect this button with play functionality of the framework
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
 
        #   set a slider covering progress of playing video, connect slider to the slot that
        # will connect it to the video playing progress
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)
 
        self.error = QLabel()
        self.error.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        #   set a buttom responsible for opening the video file that will be played
        openButton = QPushButton("Open Video")   
        openButton.setToolTip("Open Video File")
        openButton.setStatusTip("Open Video File")
        openButton.setFixedHeight(24)
        openButton.clicked.connect(self.openFile)
 
        #   considering that all previous components are just subparts of the video, it is
        # required to make a widget for the main window
        wid = QWidget(self)
        self.setCentralWidget(wid)
        
        #   there are two different parts of the media player - video section and control
        # panel. Therefore it is required to set both of them separately as layouts
        #   control panel layout
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
 
        #   combine video section with control panel in a separate layout
        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.error)
        layout.addWidget(openButton)
 
        wid.setLayout(layout)
 
        #   set for the video player where video will be shown, handle change of states
        # changes of the position where playthrough is at the moment, handle duration of
        # the video
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

        #   set media to be watched and change play state only in case if there is
        # any file chosen
        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
 
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
videoplayer = VideoPlayer()
videoplayer.resize(640, 480)
videoplayer.show()
sys.exit(app.exec_())