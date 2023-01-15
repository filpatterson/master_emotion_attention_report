import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

from abc import ABC, abstractmethod

import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


class Transformer(ABC):
    """abstract class defining main common method for all transformers that will be used for passing data
    through the pipe - transform
    """

    @abstractmethod
    def transform(self, image):
        pass


class FaceExtractor(Transformer):
    """Transformer for extracting face out of the picture

    Args:
        Transformer: basic abstract transformer class
    """

    def __init__(self, face_detection_model, offset: float = 0.06):
        """form transformer giving the model for face detection and offset that
        will be used to define box that will be cropped out of the picture

        Args:
            face_detection_model: mediapipe's model for face detection
            offset (float, optional): offset for cropping face out of the picture. 
                                    Defaults to 0.06.
        """
        self.face_detection_model = face_detection_model
        self.offset = offset

    def __crop_image__(self, image, x_min: float, y_min: float, width: float, height: float,
                       image_cols: int, image_rows: int, offset: float = 0.06):
        """crop given image using boundaries found by mediapipe face detection model with some offset
        to make sure that we covered the entire face

        Args:
            image (np.array?): image that it is required to crop
            x_min (float): min relative x coordinate of face by face detection (% between 0 and 1)
            y_min (float): min relative y coordinate of face by face detection (% between 0 and 1)
            width (float): relative size of face on x axis (% between 0 and 1)
            height (float): relative size of face on y axis (% between 0 and 1)
            image_cols (int): pixels in image on x axis
            image_rows (int): pixels in image on y axis
            offset (float, optional): relative size of offset (% between 0 and 1). Defaults to 0.06 (or 6%)

        Returns:
            np.array?: cropped image
        """
        #   crop image with some offset to make sure that entire face is covered, not only some 'basic landmarks'
        xmin, ymin = max(x_min - offset, 0), max(y_min - offset, 0)
        xmax, ymax = min(xmin + width + offset * 2, 1), min(ymin + height + offset * 2, 1)

        #   boundaries are set as relative coordinates from 0 to 1 and it is required to rescale them back
        # to image dimensions
        start_point = _normalized_to_pixel_coordinates(xmin, ymin, image_cols, image_rows)
        end_point = _normalized_to_pixel_coordinates(xmax, ymax, image_cols, image_rows)

        #   return cropped image
        return image[start_point[0]:end_point[0], start_point[1]:end_point[1]]

    def transform(self, image):
        """extract faces out of the given image

        Args:
            image (np.array?): image where faces can be found
            face_detection_model (?): mediapipe model for detecting faces
            offset (float): offset in % of how to crop image (defaults to 6% or 0.06)

        Returns:
            np.array?: images with extracted faces
        """
        #   copy original image and transform to rgb form (grayscale is not accepted),
        # get image resolution (in pixels) and then give the image for processing
        img_copy = image
        image_rows, image_cols, _ = img_copy.shape
        detection_result = self.face_detection_model.process(img_copy)

        #   if there are no faces - return None. Otherwise, go through all detected faces,
        # crop images to leave only faces
        if detection_result.detections is not None:
            for face_no, face in enumerate(detection_result.detections):
                img_copy = self.__crop_image__(img_copy,
                                               detection_result.detections[0].location_data.relative_bounding_box.xmin,
                                               detection_result.detections[0].location_data.relative_bounding_box.ymin,
                                               detection_result.detections[0].location_data.relative_bounding_box.width,
                                               detection_result.detections[0].location_data.relative_bounding_box.height,
                                               image_cols,
                                               image_rows,
                                               self.offset
                                               )
                return img_copy
        return None


class RatioRescaler(Transformer):
    """Transformer for rescaling the picture, keeping the original ratio

    Args:
        Transformer: basic abstract Transformer class
    """

    def __init__(self, target_pixels_count: int = 200):
        self.target_pixels_count = target_pixels_count

    def transform(self, image):
        """scale down the image, keeping the ratio

        Args:
            image (np.array?): image to rescale

        Returns:
            np.array?: rescaled image
        """
        image_height, image_width = image.shape[0], image.shape[1]
        if image_height >= image_width:
            scale_factor = math.ceil(self.target_pixels_count / image_height * image_height)
            img_copy = image.copy()
            img_copy = np.array(img_copy, dtype='uint8')
            resized_image = cv2.resize(img_copy, (scale_factor, scale_factor), interpolation=cv2.INTER_AREA)
            return resized_image
        if image_width > image_height:
            scale_factor = math.ceil(self.target_pixels_count / image_width * image_width)
            img_copy = image.copy()
            img_copy = np.array(img_copy, dtype='uint8')
            resized_image = cv2.resize(img_copy, (scale_factor, scale_factor), interpolation=cv2.INTER_AREA)
            return resized_image


class FaceMeshDrawer(Transformer):
    """Face Mesh drawer for drawing landmarks on the face

    Args:
        Transformer: transformer abstract class
    """

    def __init__(self, mp_face_mesh, face_mesh_images, mp_drawing, mp_drawing_styles):
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh_images = face_mesh_images
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

    def transform(self, image):
        """draw detected face landmarks on the face

        Args:
            image (np.array?): image where face will be scanned

        Returns:
            np.array?: changed image
        """
        #   copy image, make it rgb (because mediapipe face mesh works only with rgb),
        # and give image to the model
        try:
            img_copy = image
            face_mesh_results = self.face_mesh_images.process(img_copy)

            #   if face mesh found face landmarks, then show them on image
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image=img_copy,
                                            landmark_list=face_landmarks,
                                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 255),
                                                                                                thickness=1,
                                                                                                circle_radius=1),
                                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                self.mp_drawing.draw_landmarks(image=img_copy,
                                            landmark_list=face_landmarks,
                                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                            landmark_drawing_spec=None,
                                            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                return img_copy
        except:
            print('error getting faces from the picture')
            return None    


class FaceMeshCollecter(Transformer):
    """Face Mesh drawer for drawing landmarks on the face

    Args:
        Transformer: transformer abstract class
    """

    def __init__(self, face_mesh_images):
        self.face_mesh_images = face_mesh_images

    def transform(self, image):
        """draw detected face landmarks on the face

        Args:
            image (np.array?): image where face will be scanned

        Returns:
            np.array?: changed image
        """
        img_copy = image
        face_mesh_results = self.face_mesh_images.process(img_copy)
        
        if face_mesh_results.multi_face_landmarks is not None:
            for face in face_mesh_results.multi_face_landmarks:
                x_num = np.array([landmark.x for landmark in face.landmark])
                y_num = np.array([landmark.y for landmark in face.landmark])
                z_num = np.array([landmark.z for landmark in face.landmark])
                return np.vstack([x_num, y_num, z_num]).T
        return None


class ImageReader(Transformer):
    def __init__(self):
        pass

    def transform(self, image):
        try:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            return img
        except:
            print('error reading file ' + image)
            return None


class Pipeline:
    """pipeline that will collect all transforms and pass data through it
    """

    def __init__(self, dict_of_transformers: dict = None):
        """give dictionary of transformers

        Args:
            dict_of_transformers (dict[str, Transformer], optional): transformers 
                                                with their names. Defaults to None.
        """
        self.transformers = dict_of_transformers

    def set_pipeline(self, dict_of_transformers: dict):
        """set new dictionary of transformers

        Args:
            dict_of_transformers (dict[str, Transformer]): transformers and their names
        """
        self.transformers = dict_of_transformers

    def add(self, transformer_name: str, transformer: Transformer):
        """add new transformer with name to the transformers dict

        Args:
            transformer_name (str): name of transformer
            transformer (Transformer): transformer object
        """
        if self.transformers.get(transformer_name, None) is not None:
            print('typed transformer already exists, check name once again')
            return
        self.transformers[transformer_name] = transformer

    def remove(self, transformer_name: str):
        """remove transformer out of dictionary by name

        Args:
            transformer_name (str): name of transformer to delete
        """
        if self.transformers.get(transformer_name, None) is None:
            print('No such transformer, are you sure about name?')
            return
        del self.transformers[transformer_name]

    def pop(self, transformer_name: str):
        """remove and return transformer by name

        Args:
            transformer_name (str): name of transformer to pop

        Returns:
            Transformer: requested transformer
        """
        if self.transformers.get(transformer_name, None) is None:
            print('No such transformer, are you sure about name?')
            return
        return self.transformers.pop(transformer_name)

    def process(self, image):
        """pass given image through specified transformers

        Args:
            image (np.array?): image to process

        Returns:
            np.array?: final image
        """
        img_copy = image
        for transformer_name, transformer in zip(self.transformers.keys(), self.transformers.values()):
            img_copy = transformer.transform(img_copy)
            if img_copy is None:
                return img_copy
        return img_copy


def show_18_samples(df: pd.DataFrame, column_to_show: str):
    """show first 18 images out of given dataframe with specified column name

    Args:
        df (pd.DataFrame): dataframe with images
        column_to_show (str): column containing images to show
    """
    ROWS_COUNT = 6
    COLUMNS_COUNT = 3
    fig, axs = plt.subplots(ROWS_COUNT, COLUMNS_COUNT, figsize=(7, 14))
    for row in range(ROWS_COUNT):
        for column in range(COLUMNS_COUNT):
            if df.loc[(row * COLUMNS_COUNT) + column, column_to_show] is None:
                continue
            axs[row, column].imshow(df.loc[(row * COLUMNS_COUNT) + column, column_to_show].copy())
    plt.tight_layout()
    plt.show()


def read_image(path: list):
    """reads images in given paths in form of a grayscale

    Args:
        paths (list): string-formatted paths to images

    Returns:
        np.array?: image
    """
    image = cv2.imread('Manually_Annotated_Images\\' + path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return image
