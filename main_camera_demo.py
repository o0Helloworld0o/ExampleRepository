import sys
import cv2
import numpy as np

sys.path.append('../../GitHub_Face_Toolkit/Peppa_Pig_Face_Engine')
from lib.core.api.facer import FaceAna
from face_alignment.align_trans import get_reference_facial_points, warp_and_crop_face



facer = FaceAna()

landmarks_index = [68, 69, 30, 48, 54]  # 68关键点下的5点



crop_version = 'v2'
crop_size = 128
scale = crop_size / 112.0   # 相对于标准尺寸112，放大的倍数


reference = get_reference_facial_points(default_square=True)

if crop_version == 'v2':
    reference = (reference - 112.0/2) * 1.20 + 112.0/2
    reference[:, 1] = reference[:, 1] + -20

reference = reference * scale





cap = cv2.VideoCapture(0)   # 默认w=640, h=480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)





while True:
    ret, img_raw = cap.read()
    img_h, img_w = img_raw.shape[:2]


    img_rgb = img_raw[...,::-1]
    bboxes, landmarks, states = facer.run_single_image(img_rgb)



    whiteboard = np.ones([img_h, img_h, 3])
    whiteboard = (whiteboard * 255).astype(np.uint8)



    if len(bboxes) == 0:
        display_img = np.concatenate([img_raw, whiteboard], axis=1)

    else:
        # 添加2个点，分别表示左右眼中心
        left_eye = landmarks[:, range(36, 42)].mean(axis=1, keepdims=True)
        right_eye = landmarks[:, range(42, 48)].mean(axis=1, keepdims=True)
        landmarks = np.concatenate([landmarks, left_eye, right_eye], axis=1)


        face_index = 0
        bbox = bboxes[face_index].astype(int)
        facial5points = landmarks[face_index, landmarks_index]
        lmk = landmarks[face_index]

        warped_face, tfm = warp_and_crop_face(img_raw, facial5points, reference, (crop_size,)*2)




        W, b = tfm.T[:2], tfm.T[2]
        new_lmk = lmk @ W + b



        lmk_draw = warped_face.copy()
        for p in new_lmk.astype(int):
            cv2.circle(lmk_draw, (p[0], p[1]), radius=2, color=(0, 255, 0), thickness=-1)




        
        whiteboard[:crop_size, :crop_size] = warped_face
        whiteboard[:crop_size, crop_size:2*crop_size] = lmk_draw



        text = 'img_raw: w={}, h={}'.format(img_w, img_h)
        cv2.putText(whiteboard, text, (10, crop_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        text = 'img_face: w={}, h={}'.format(bbox[2] - bbox[0], bbox[3] - bbox[1])
        cv2.putText(whiteboard, text, (10, crop_size + 20 * 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)




        display_img = np.concatenate([img_raw, whiteboard], axis=1)






    cv2.imshow('frame', display_img)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break


