import os
import cv2
import dlib
import time
import argparse
import numpy as np
from imutils import video

DOWNSAMPLE_RATIO = 4


def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def main():
    os.makedirs('original', exist_ok=True)
    os.makedirs('landmarks', exist_ok=True)

    cap = cv2.VideoCapture(args.filename)
    fps = video.FPS().start()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        black_image = np.zeros(frame.shape, np.uint8)

        t = time.time()

        # Perform if there is a face detected
        if len(faces) == 1:
            for face in faces:
                detected_landmarks = predictor(gray, face).parts()
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

                jaw = reshape_for_polyline(landmarks[0:17])
                left_eyebrow = reshape_for_polyline(landmarks[22:27])
                right_eyebrow = reshape_for_polyline(landmarks[17:22])
                nose_bridge = reshape_for_polyline(landmarks[27:31])
                lower_nose = reshape_for_polyline(landmarks[30:35])
                left_eye = reshape_for_polyline(landmarks[42:48])
                right_eye = reshape_for_polyline(landmarks[36:42])
                outer_lip = reshape_for_polyline(landmarks[48:60])
                inner_lip = reshape_for_polyline(landmarks[60:68])

                color = (255, 255, 255)
                thickness = 3

                cv2.polylines(black_image, [jaw], False, color, thickness)
                cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
                cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
                cv2.polylines(black_image, [nose_bridge], False, color, thickness)
                cv2.polylines(black_image, [lower_nose], True, color, thickness)
                cv2.polylines(black_image, [left_eye], True, color, thickness)
                cv2.polylines(black_image, [right_eye], True, color, thickness)
                cv2.polylines(black_image, [outer_lip], True, color, thickness)
                cv2.polylines(black_image, [inner_lip], True, color, thickness)

            # Display the resulting frame
            count += 1
            print(count)
            cv2.imwrite("original/{}.png".format(count), frame)
            cv2.imwrite("landmarks/{}.png".format(count), black_image)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if count == args.number:  # only take 400 photos
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No face detected")

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    parser.add_argument('--num', dest='number', type=int, help='Number of train data to be created.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
