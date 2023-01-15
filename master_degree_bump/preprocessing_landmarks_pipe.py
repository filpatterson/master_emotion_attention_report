import pandas as pd
from transformers import Pipeline, FaceExtractor, RatioRescaler, ImageReader, FaceMeshCollecter

import mediapipe as mp
import cv2

import math
import multiprocessing
import time
import os
import pickle

from os import listdir
from os.path import isfile, join

JOBS_COUNT = 4
CHUNK_SIZE = 20000


def landmarks_extraction_job(df_part):
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                               min_detection_confidence=0.5)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                min_detection_confidence=0.75)
    pipeline_dict = {
        'image_reader': ImageReader(),
        # 'face_extractor': FaceExtractor(face_detection, offset=0.15),
        # 'ratio_rescaler': RatioRescaler(target_pixels_count=200),
        'mesh_collecter': FaceMeshCollecter(face_mesh)
    }

    pipeline = Pipeline(pipeline_dict)
    print('start processing part ' + str(os.getpid()))
    df_part['landmarks'] = [pipeline.process(path) for path in df_part.path]
    print('ended processing part ' + str(os.getpid()))
    with open(str(os.getpid()) + '_landmarks_valence_arousal.pkl', 'wb') as handle:
        pickle.dump(df_part, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # df_part.to_csv(str(os.getpid()) + '_landmarks.csv')


def process_images(df):
    jobs = []

    # for each of the chunks for sub-chunks that will be sent to child processes
    sub_chunk_size = math.ceil(len(df) / JOBS_COUNT)
    for sub_i in range(JOBS_COUNT):
        if sub_i == JOBS_COUNT - 1:
            process = multiprocessing.Process(target=landmarks_extraction_job,
                                                args=(df[
                                                        sub_i * sub_chunk_size: len(df) - 1
                                                    ].copy(),))
        else:
            process = multiprocessing.Process(target=landmarks_extraction_job,
                                                args=(df[
                                                        sub_i * sub_chunk_size: (sub_i + 1) * sub_chunk_size
                                                    ].copy(),))
        jobs.append(process)
        process.start()

    for job in jobs:
        job.join()
    for job in jobs:
        job.kill()
    for job in jobs:
        del job
    del jobs
    
    df = df[0:0]
    del df


if __name__ == '__main__':
    #   affect net manual dataset read
    affect_net_manual_df = pd.read_csv('Manually_Annotated_file_lists/training.csv')
    affect_net_manual_df.drop(columns=['face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks'],
                              inplace=True)
    affect_net_manual_df.rename(columns={'subDirectory_filePath': 'path', 'expression': 'emotion'},
                                inplace=True)
    affect_net_manual_df['path'] = ['Manually_Annotated_Images/' + path for path in affect_net_manual_df.path]

    #   affect net short dataset read
    # affect_net_short_df = pd.read_csv('affect-net/labels.csv')
    # affect_net_short_df.rename(columns={'pth': 'path', 'label': 'emotion'}, inplace=True)
    # affect_net_short_df['path'] = ['affect-net/' + path for path in affect_net_short_df.path]

    #   raf db dataset read
    # raf_df = pd.read_csv('raf-db/list_patition_label.txt', sep='\t', header=None)
    # raf_df.columns = ['path', 'emotion']
    # raf_df['path'] = ['raf-db/aligned/' + path[:-4] + '_aligned.jpg' for path in raf_df.path]

    #   ck+ dataset read
    # ck_dict = {}
    # ck_df = pd.DataFrame()
    # sub_dirs = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    # for emotion in sub_dirs:
    #     path = 'ck_complete/' + emotion
    #     files = [path + '/' + f for f in listdir(path) if isfile(join(path, f))]
    #     for file in files:
    #         ck_dict[file] = emotion
    # ck_df['path'] = ck_dict.keys()
    # ck_df['emotion'] = ck_dict.values()

    #   standardize all datasets to the same emotions representations
    # raf_df.replace({7: 0, 4: 1, 5: 2, 1: 3, 2: 4, 3: 5, 6: 6},
    #                inplace=True)
    # affect_net_short_df.replace({'surprise': 3, 'anger': 6, 'fear': 4, 'disgust': 5, 'sad': 2, 'neutral': 0,
    #                              'contempt': 7, 'happy': 1},
    #                             inplace=True)
    # ck_df.replace({'anger': 6, 'contempt': 7, 'disgust': 5, 'fear': 4, 'happy': 1, 'sadness': 2, 'surprise': 3},
    #               inplace=True)

    #   forming final big dataset
    # affect_net_total_df = pd.concat([affect_net_short_df, affect_net_manual_df])
    # affect_net_with_raf_df = pd.concat([affect_net_total_df, raf_df])
    # affect_raf_ck_df = pd.concat([affect_net_with_raf_df, ck_df])
    # affect_raf_ck_df.reset_index(inplace=True)
    # affect_raf_ck_df.drop(columns=['index'], inplace=True)
    
    chunks_count = math.ceil(len(affect_net_manual_df) / CHUNK_SIZE)
    
    # for i in range(chunks_count):
    #     start_cycle_time = time.time()
    #     if i != chunks_count - 1:
    #         process_images(affect_raf_ck_df.loc[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].copy())
    #     else:
    #         process_images(affect_raf_ck_df.loc[i * CHUNK_SIZE:len(affect_raf_ck_df)].copy())
    #     print('Processing cycle ' + str(i) + ' took ' + str(time.time() - start_cycle_time) + ' sec.')
        
    for i in range(chunks_count):
        start_cycle_time = time.time()
        if i != chunks_count - 1:
            process_images(affect_net_manual_df.loc[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].copy())
        else:
            process_images(affect_net_manual_df.loc[i * CHUNK_SIZE:len(affect_net_manual_df)].copy())
        print('Processing cycle ' + str(i) + ' took ' + str(time.time() - start_cycle_time) + ' sec.')