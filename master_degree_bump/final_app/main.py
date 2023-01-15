from video_player import *
from webcam_stream import *
from mp_facemesh_getter import *
from point_net import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = get_base_point_net_model()
    model = point_net_weights_load(model, 'models/v3_point_net/')
    webcam_stream = App(get_mediapipe_facemesh_drawer(), get_mediapipe_facemesh_collecter(), model)
    videoplayer = VideoPlayer(webcam_stream)
    videoplayer.resize(1920, 760)
    videoplayer.show()
    sys.exit(app.exec_())