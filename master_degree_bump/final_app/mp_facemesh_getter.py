import keras
from tensorflow.keras import layers
import mediapipe as mp
from transformers import FaceMeshCollecter, FaceMeshDrawer, Transformer

def get_mediapipe_facemesh_drawer():
    """form a mediapipe facemesh drawer that will show how mediapipe scans faces
    :return: Mediapipe Facemesh Drawer model that will draw 3D scan over the original face
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True,
                                             max_num_faces=1,
                                             min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return FaceMeshDrawer(mp_face_mesh, face_mesh_images, mp_drawing, mp_drawing_styles)

def get_mediapipe_facemesh_collecter():
    return FaceMeshCollecter(mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.75))
