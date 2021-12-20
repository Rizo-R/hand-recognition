from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse


import mediapipe as mp
import numpy as np
import csv
import json
from pathlib import Path

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_true = np.loadtxt(open("labels_true.csv", "rb"),
                         delimiter=",", skiprows=0).astype(int)

LANDMARK_THRESHOLD = 10
TOP_THRESHOLD = 0.35
AREA_THRESHOLD = 0.2
ENSEMBLE = True


def calculate_statistics(pred_labels, true_labels):
    # Count true positives, false positives, true negatives, false negatives
    assert pred_labels.shape == true_labels.shape
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(pred_labels.shape[0]):
        if pred_labels[i, 1] == true_labels[i, 1] == 1:
            tp += 1
        elif pred_labels[i, 1] == true_labels[i, 1] == 0:
            tn += 1
        elif pred_labels[i, 1] == 1 and true_labels[i, 1] == 0:
            fp += 1
        else:
            fn += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    res = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    return res


def make_predictions(top_threshold=0.3, area_threshold=0.04, ensemble=False):
    detection_graph, sess = detector_utils.load_inference_graph()
    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    frame_list = []
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        try:
            boxes, scores = detector_utils.detect_objects(image_np,
                                                          detection_graph, sess)
        except:
            print("BROKE AT FRAME %i." % num_frames)
            break

        landmark_list = []
        if ensemble:
            with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
                results = hands.process(image_np)
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            landmark_list.append((landmark.x, landmark.y))

        # NEW: Delete boxes above the top threshold or smaller than the area threshold
        new_boxes = []
        new_scores = []
        for i in range(len(boxes)):
            top, left, bottom, right = boxes[i]
            # Check if area is large enough
            area = (bottom-top)*(right-left)
            # NEW: Check if box contains at least 10 landmarks
            landmark_count = 0
            for l_x, l_y in landmark_list:
                if left <= l_x <= right and top <= l_y <= bottom:
                    landmark_count += 1
            if landmark_count >= 10 or (top > top_threshold and area > area_threshold):
                new_boxes.append(boxes[i])
                new_scores.append(scores[i])
        boxes = np.asarray(new_boxes)
        scores = np.asarray(new_scores)

        frame_boxes = detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

        if len(frame_boxes) > 0:
            frame_list.append([num_frames, 1])
        else:
            frame_list.append([num_frames, 0])

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))

    print("Finished labeling %i frames!" % num_frames)
    print("Elapsed time:", elapsed_time)
    return np.asarray(frame_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    best_prec = 0
    best_prec_val = None
    best_acc = 0
    best_acc_val = None
    best_rec = 0
    best_rec_val = None

    labels_pred = make_predictions(TOP_THRESHOLD, AREA_THRESHOLD, ENSEMBLE)
    if ENSEMBLE:
        save_dir = "./results_ensemble"
    else:
        save_dir = "./results"

    # Make a directory if it doesn't exist; save the predictions
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(save_dir+"/T%s_A%s.csv" % (TOP_THRESHOLD, AREA_THRESHOLD),
               labels_pred, fmt="%i", delimiter=",")

    res = calculate_statistics(labels_pred, labels_true)

    print("Best accuracy:", best_acc, best_acc_val)
    print("Best precision:", best_prec, best_prec_val)
    print("Best recall:", best_rec, best_rec_val)
