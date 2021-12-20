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

PATH = "./combine_neck_pilot.mp4"
SCORE_THRESHOLD = 0.2
LANDMARK_THRESHOLD = 10


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
    cap = cv2.VideoCapture(PATH)
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
            num_hands_detect, SCORE_THRESHOLD, scores, boxes, im_width, im_height, image_np)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

        if len(frame_boxes) > 0:
            frame_list.append([num_frames, 1])
        else:
            frame_list.append([num_frames, 0])

    print("Finished labeling %i frames!" % num_frames)
    print("Elapsed time:", elapsed_time)
    return np.asarray(frame_list)


def load_predictions(top_threshold_list, area_threshold_list, ensemble=False):
    best_prec = 0
    best_prec_val = None
    best_acc = 0
    best_acc_val = None
    best_rec = 0
    best_rec_val = None
    for top in top_threshold_list:
        for area in area_threshold_list:
            labels_pred = np.loadtxt(
                open("./results/T%s_A%s.csv" % (top, area), "rb"), delimiter=",", skiprows=0)
            res = calculate_statistics(labels_pred, labels_true)
            if res["accuracy"] > best_acc:
                best_acc = res["accuracy"]
                best_acc_val = "T=%s,A=%s" % (top, area)
            if res["precision"] > best_prec:
                best_prec = res["precision"]
                best_prec_val = "T=%s,A=%s" % (top, area)
            if res["recall"] > best_rec:
                best_rec = res["recall"]
                best_rec_val = "T=%s,A=%s" % (top, area)
    print("Best accuracy:", best_acc, "Best accuracy values:", best_acc_val)
    print("Best precision:", best_prec,
          ". Best precision values:", best_prec_val)
    print("Best recall:", best_rec, "Best recall values:", best_rec_val)
    return


if __name__ == '__main__':
    # top_threshold_list = ["0", "0.2", "0.25", "0.3", "0.35"]
    # area_threshold_list = ["0", "0.02", "0.04", "0.1", "0.2"]
    top_threshold_list = ["0.2"]
    area_threshold_list = ["0.04"]
    # Change this if want to not use ensembling
    ensemble = False

    best_prec = 0
    best_prec_val = None
    best_acc = 0
    best_acc_val = None
    best_rec = 0
    best_rec_val = None

    results = []
    for top_count in range(len(top_threshold_list)):
        row = []
        for area_count in range(len(area_threshold_list)):
            top = top_threshold_list[top_count]
            area = area_threshold_list[area_count]
            print("Trying T=%s, A=%s" % (top, area))
            labels_pred = make_predictions(float(top), float(area))
            if ensemble:
                save_dir = "./results_ensemble"
            else:
                save_dir = "./results"
            # Make a directory if it doesn't exist; save the predictions
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.savetxt(save_dir+"/T%s_A%s.csv" % (top, area),
                       labels_pred, fmt="%i", delimiter=",")

            res = calculate_statistics(labels_pred, labels_true)
            row.append(res)
            if res["accuracy"] > best_acc:
                best_acc = res["accuracy"]
                best_acc_val = "T=%s,A=%s" % (top, area)
            if res["precision"] > best_prec:
                best_prec = res["precision"]
                best_prec_val = "T=%s,A=%s" % (top, area)
            if res["recall"] > best_rec:
                best_rec = res["recall"]
                best_rec_val = "T=%s,A=%s" % (top, area)
        results.append(row)

    print("Best accuracy:", best_acc, best_acc_val)
    print("Best precision:", best_prec, best_prec_val)
    print("Best recall:", best_rec, best_rec_val)

    # Save results
    if ensemble:
        res_path = './results_ensemble.csv'
    else:
        res_path = './results.csv'
    with open('./results.csv', 'w') as fout:
        json.dump(results, fout)

    load_predictions(top_threshold_list, area_threshold_list, False)
