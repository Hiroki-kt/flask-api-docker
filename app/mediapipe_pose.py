import gc
import glob
import os
import sys

from argparse import ArgumentParser
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from memory_profiler import profile
import numpy as np
import sympy as sm
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)  # gray


class MPTools:
    @staticmethod
    def crop_box(box, image):
        return image[box[1]:box[3], box[0]:box[2], :]

    @staticmethod
    def get_box_size(separate_num=(4, 3), dist_image_size=(192, 192)):
        '''
        default separate_num = (4, 3), image_size = (192, 192)
        if you make '12' boxes, you can make square images.
        [args]
        separate_num(list): box separate num [w,h]
        image_size(list): image_size [w, h]
        [return]
        box_size(list): [box_w,box_h]
        '''

        box_w = int(dist_image_size[0] / separate_num[0])
        box_h = int(dist_image_size[1] / separate_num[1])

        return [box_w, box_h]

    @staticmethod
    def get_landmark_box(point, box_size):
        '''
        [args]
        point(list): box center [x, y]
        box_size(list): one box size [w, h]
        [return]
        landmark_box(ndarry):
        landmark_box[i] = [x1, y1, x2, y2]
        x1, y2: left up
        x2, y2: right down
        '''
        x1 = int(point[0] - (box_size[0] / 2))
        y1 = int(point[1] - (box_size[1] / 2))
        x2 = x1 + box_size[0]
        y2 = y1 + box_size[1]

        return [x1, y1, x2, y2]

    @staticmethod
    def get_midpoint(x, y):
        # midpoint = (x + y) / 2
        # print(midpoint)
        return ((x + y) / 2).astype(np.int)

    @staticmethod
    def make_white_noise_box(imagesize):
        # ボックスをホワイトノイズで作成(デフォルト)
        random_array = bytearray(
            os.urandom(imagesize[0] * imagesize[1]))
        np_array = np.array(random_array)  # 1D 乱数

        np_img = np_array.reshape(imagesize)
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        # pil
        # white_noise_img = Image.fromarray(np.uint8(np_img))
        return rgb_img


class MediaPipePose(MPTools):
    def __init__(self, dryrun=False, segmentation=True):
        super().__init__()
        self.segmentation = segmentation
        self.dryrun = dryrun

    def get_body_center(self, image, results, vector_center=True):
        '''
        Auth: hiroki-kt
        [args]
        image: numpy ndarray(opencv)
        results: mediapipelist
        vector_center: default True, if you want vectort center, this value is 'True'. If you want physical center, this value is 'False'
        [return]
        center: numpy ndarray [2,]
        body_box:numpy ndarray [4, 2]
            0: left_shoulder = results.pose_landmarks.landmark[11]
            1: right_shoulder = results.pose_landmarks.landmark[12]
            2: left_hip = results.pose_landmarks.landmark[23]
            3: right_hip = results.pose_landmarks.landmark[24]
        '''
        annotated_image = image.copy()
        landmark_id = [11, 12, 23, 24]
        body_box = self.get_pose_positions(
            annotated_image, results, landmark_id)
        if vector_center:
            # ベクトルの重心を求めたい場合
            center = (np.sum(body_box, axis=0) / 4).astype(np.int32)
        else:
            # 物理的重心を求めたい場合
            s1 = sm.Segment(body_box[0], body_box[3])
            s2 = sm.Segment(body_box[1], body_box[2])
            center = np.array(s1.intersection(s2)).astype(np.int32).reshape(2,)
        # img = cv2.circle(annotated_image, center, 5, (0, 0, 255))
        # img = cv2.circle(img, body_center, 5, (0, 255, 0))
        # cv2.imwrite("test-body-center.png", img)
        return center, body_box

    def get_pose_positions(self, image, results, landmark_id):
        '''
        Auth: hiroki-kt
        [args]
        image: numpy ndarray(opencv)
        results: mediapipelist
        landmark_id(list): [0 ~ 32]
        you chose the landmark ids from mediapipe hp: https://google.github.io/mediapipe/solutions/pose
        [return]
        landmark_postions:numpy ndarray [n, 2]
        n = len(landmark_id), you can get landmarks's x, and y postion.
        '''
        _landmark_postions = np.zeros((len(landmark_id), 2), dtype=np.int)
        if results.pose_landmarks is None:
            print("error")
            return results.pose_landmarks
        else:
            for i, ids in enumerate(landmark_id):
                pos = results.pose_landmarks.landmark[ids]
                _landmark_postions[i, 0] = int(pos.x * image.shape[1])
                _landmark_postions[i, 1] = int(pos.y * image.shape[0])
            return _landmark_postions

    # @profile

    def get_born(self, image_file, no_img=False):
        '''
        Auth: hiroki-kt
        [args]
        image_file: str, file path or ndarray
        [return]
        results: mediapipelist, pose born list and value is float (0 ~ 1)
        '''
        # print(image_file)
        if type(image_file) is str:
            cv2_img = cv2.imread(image_file)
        else:
            cv2_img = image_file

        # image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        # オブジェクト削除
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=self.segmentation,
                min_detection_confidence=0.5) as pose:
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            pose_data = pose.process(img)
            # print(results.pose_landmarks)
        if no_img:
            return None, pose_data
        else:
            return cv2_img, pose_data

    def crop_body(self, image, results):
        '''
        test for crop the body by using
        '''
        annotated_image = image.copy()
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        x1 = int(right_shoulder.x * image.shape[1])
        y1 = int(right_shoulder.y * image.shape[0])
        x2 = int(left_hip.x * image.shape[1])
        y2 = int(left_hip.y * image.shape[0])
        # print(x1 * image.shape[1], y1 * image.shape[0],
        #       x2 * image.shape[1], y2 * image.shape[0])
        img = cv2.rectangle(annotated_image, (x1, y1),
                            (x2, y2), (255, 255, 0), 3)
        cv2.imwrite("test-box.png", img)

    def write_born(self, image, results, output, landmark_on=True, crop=None):
        '''
        test for save image by drawing the born
        '''
        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        if self.segmentation:
            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        if landmark_on:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if crop is not None:
            crop = crop.astype(np.int)
            annotated_image = annotated_image[crop[1]                                              : crop[3], crop[0]: crop[2], :]
        if self.dryrun:
            print(output)
            out_fname = 'mp-test.jpg'
            cv2.imwrite(out_fname, annotated_image)
            print('---[save sample image]---')
            # sys.exit()
        else:
            cv2.imwrite(output, annotated_image)
        # print('saved')
        # Plot pose world landmarks.
        # mp_drawing.plot_landmarks(
        #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    def crop_landmark(self, image, results, output, random_crop=False):
        '''
        [args]
        image: numpy ndarray(opencv)
        results: mediapipelist
        [return]
        landmark_postions:
        landmark_boxes:
        '''
        # print(f'image size:{image.shape}')
        annotated_image = image.copy()
        points = [11, 12, 23, 24, 13, 14, 25, 26]

        '''
        0(11): left_shoulder
        1(12): right_shoulder
        2(23): left_hip
        3(24): right_hip
        4(13): left_elbow
        5(14): right_elbow
        6(25): left_knee
        7(26): right_knee
        '''
        if results.pose_landmarks is None:
            print("Error: can't detect human born landmark point")
            return None
        landmark_positions = np.array(
            self.get_pose_positions(image, results, points))
        # print(output)
        body_box = landmark_positions[:4]

        append_list = []
        append_list.append((np.sum(body_box, axis=0) /
                           4).astype(np.int32))  # center
        append_list.append(self.get_midpoint(body_box[0], body_box[1]))  # neck
        append_list.append(self.get_midpoint(
            body_box[0], body_box[2]))  # left_flank
        append_list.append(self.get_midpoint(
            body_box[1], body_box[3]))  # right_flank
        # print(landmark_positions.shape)
        for i in append_list:
            landmark_positions = np.append(
                landmark_positions, i.reshape(1, -1), axis=0)
        # print(landmark_positions.shape)
        '''
        0(11): left_shoulder
        1(12): right_shoulder
        2(23): left_hip
        3(24): right_hip
        4(13): left_elbow
        5(14): right_elbow
        6(25): left_knee
        7(26): right_knee
        8(): body center
        9(): neck
        10(): left_flank
        11(): right_flank
        '''
        landmark_boxes = np.zeros((landmark_positions.shape[0], 4), np.int32)
        image_size = 192
        if random_crop:
            size_list = [168, 192, 216, 240]
            image_size = size_list[np.random.randint(4)]
        dist_img = np.zeros((image_size, image_size, 3), np.int32)
        # ↓ default separate_num = (4, 3), image_size = (192, 192)
        box_size = self.get_box_size(dist_image_size=(image_size, image_size))
        # print(box_size)
        _w = box_size[0]
        _h = box_size[1]
        # print(dist_img.shape)
        # print(f'{output}')
        points_num = landmark_positions.shape[0]
        for i, point in enumerate(landmark_positions):
            if random_crop:
                point += np.random.randint(-15, 15, 2)
            _box = self.get_landmark_box(point, box_size)
            n = i // 3
            m = i % 3
            # print(f'point:{point}/{i}')
            # print(f'i:{i}, n:{n}, m:{m}')
            _crop = self.crop_box(_box, annotated_image)
            # print(f'crop:{_crop.shape}')
            if _crop.shape != (_h, _w, 3):
                # print(f"miss:id:{i}/crop:{_crop.shape}/point:{point}")
                dist_img[int(m * _h):int(m * _h) + _h, int(n * _w)
                             :int(n * _w) + _w, :] = self.make_white_noise_box((_h, _w))
            else:
                dist_img[int(m * _h):int(m * _h) + _h, int(n * _w)
                             :int(n * _w) + _w, :] = _crop
            landmark_boxes[i, :] = _box
            # print(f'======={i}=======')
            # print(f'box:{_box}')
            # print(f'point:{point}')
        if self.dryrun:
            print(output)
            out_fname = 'landmark-crop-test.jpg'
            cv2.imwrite(out_fname, dist_img)
            print('---[save sample image]---')
            # sys.exit()
        else:
            if output is None:
                return dist_img
            else:
                cv2.imwrite(output, dist_img)
        return landmark_positions, landmark_boxes


def random_crop_test(image_file_path):
    output_dir = "./random-crop-test"
    os.makedirs(output_dir, exist_ok=True)
    mpp = MediaPipePose(dryrun=False, segmentation=False)
    for i in range(5):
        image, results = mpp.get_born(image_file_path)
        output_file = f'{output_dir}/test-{i}.png'
        mpp.crop_landmark(image, results, output_file, random_crop=True)


if __name__ == '__main__':
    mpp = MediaPipePose(dryrun=True, segmentation=False)
    image_fname = 'western/CAMR-01/R_039842784_A_031_comp[1]_0.jpg'
    image_dir = '/mnt/dataspace/docker-space/pytorch-kata/hosho/crop-single'
    # image_fname = 'kimono/39_21KH-01/IMG_0617IMG_0617.jpg'
    # image_dir = '/mnt/dataspace/docker-space/pytorch-kata/hosho/label_sample'
    # output = 'test'
    image_fpath = f'{image_dir}/{image_fname}'
    print(image_fpath)
    # image, results = mpp.get_born(image_fpath)
    # mpp.write_born(image, results, output)
    # result = mpp.crop_landmark(image, results, output)
    # if result is None:
    #     print("error")
    random_crop_test(image_fpath)
