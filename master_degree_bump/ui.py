from tkinter import *

import keras
from PIL import ImageTk, Image
import cv2

from point_net_api import *
from tensorflow.keras import layers
import mediapipe as mp
from transformers import FaceMeshCollecter, FaceMeshDrawer, Transformer

first_degree, second_degree, third_degree, fourth_degree = 64, 128, 256, 512


def initialize_mediapipe_mesh():
    return FaceMeshCollecter(mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                             min_detection_confidence=0.75))


def initialized_mediapipe_mesh_drawer():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True,
                                             max_num_faces=1,
                                             min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return FaceMeshDrawer(mp_face_mesh, face_mesh_images, mp_drawing, mp_drawing_styles)


def point_net_initialization(first_degree: int, second_degree: int,
                             third_degree: int, fourth_degree: int):
    inputs = keras.Input(shape=(468, 3))

    x = tnet(inputs, 3, first_degree, second_degree, third_degree, fourth_degree)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)

    x = tnet(x, 32, first_degree, second_degree, third_degree, fourth_degree)
    x = conv_bn(x, 32)
    x = conv_bn(x, first_degree)
    x = conv_bn(x, third_degree)

    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, second_degree)
    x = layers.Dropout(0.4)(x)
    x = dense_bn(x, first_degree)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(8, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    return model


def point_net_weights_load(model: keras.Model, path: str):
    model.load_weights(path)
    return model


class VideoStream:
    def __init__(self, model: keras.Model, mesh_collecter: Transformer, mesh_drawer: Transformer):
        self.prediction_model = model
        self.mesh_collecter = mesh_collecter
        self.mesh_drawer = mesh_drawer

    def video_stream(self):
        _, frame = stream.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = self.mesh_collecter.transform(cv2image)
        landmarked_image = self.mesh_drawer.transform(cv2image)
        if landmarks is not None:
            print(self.prediction_model.predict(landmarks.reshape(1, 468, 3)))

        img = Image.fromarray(cv2image)
        if landmarked_image is not None:
            img = Image.fromarray(landmarked_image)
        imgtk = ImageTk.PhotoImage(image=img)

        label_main.imgtk = imgtk
        label_main.configure(image=imgtk)
        label_main.after(1, self.video_stream)


if __name__ == '__main__':
    face_mesh_collecter = initialize_mediapipe_mesh()
    face_mesh_drawer = initialized_mediapipe_mesh_drawer()
    model = point_net_initialization(first_degree, second_degree, third_degree, fourth_degree)
    model = point_net_weights_load(model, 'models/better_v1_1_point_net/')
    video_stream = VideoStream(model, face_mesh_collecter, face_mesh_drawer)

    #   main window of the Tkinter
    root = Tk()

    app = Frame(root, bg='white')
    app.grid()

    label_main = Label(app)
    label_main.grid()

    #   enable cam stream
    stream = cv2.VideoCapture(0)

    video_stream.video_stream()
    root.mainloop()
