"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from collections import deque

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, normalize_box, unnormalize_box

np.random.seed(0)


def linear_assignment(cost_matrix: np.ndarray):
    """ solves the linear assignment problem minimizing the cost

    :param cost_matrix: contains -iou values for each possible pair
    :return: a two-dimensional array containing the correspondences
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray):
    """ from SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]

    :param bb_test: batched predicted bounding boxes in the form [x1,y1,x2,y2]
    :param bb_gt: batched detected bounding boxes in the form [x1,y1,x2,y2]
    :return: matrix which contains iou values for each possible pair
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def associate_detections_to_trackers(detections: np.ndarray, trackers: np.ndarray, iou_threshold=0.3):
    """ Assigns detections to tracked object (both represented as bounding boxes)

    :param detections: detections from the current track in the form [x1,y1,x2,y2,score]
    :param trackers:  list of predicted bounding boxes
    :param iou_threshold: minimum iou threshold to be matched
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    # matches predictions to detections
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # determine unmatched detections
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # determine unmatched predictions (unmatched tracks)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Track(object):
    """
    Represents a tracked object in a sequence
    """
    count = 0

    def __init__(self, dets):
        dets = box_xyxy_to_cxcywh(torch.tensor(dets[:4])).squeeze()
        delta = dets - dets
        self.id = Track.count
        Track.count += 1
        self.last_pos = dets
        self.seq_pos = deque(dets, maxlen=10)
        self.time_since_update = 0
        self.hit_streak = 0

        self.seq_features = deque(maxlen=10)
        self.seq_features.append(torch.concatenate([dets, delta], dim=0))

    def update(self, dets: np.ndarray):
        """ updates the position of the tracked object

        :param dets: bounding box and score in the form [x1,y1,x2,y2,score]
        """
        dets = box_xyxy_to_cxcywh(torch.tensor(dets[:4])).squeeze()
        delta = dets - self.last_pos
        self.last_pos = dets
        self.seq_pos.append(dets)
        self.seq_features.append(torch.concatenate([dets, delta], dim=0))
        self.time_since_update = 0
        self.hit_streak += 1

    def get_last_pos(self):
        """ getter for the current position of the tracked object

        :return: current position in the form [x_center,y_center,w,h] as a torch.tensor
        """
        return self.last_pos

    def get_seq_features(self):
        """ getter for the n previous positions of the tracked object

        :return: n previous positions in the form [x_center,y_center,w,h] as a torch.tensor
        """
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return torch.stack(list(self.seq_features)).unsqueeze(0)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """ sets key parameters for SORT

        :param max_age: how many frames an unmatched track persists
        :param min_hits: how many successful matches are needed to initialize a tracker
        :param iou_threshold: minimum iou threshold to be matched
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), image_size=(0, 0), motion_model=None):
        """ this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).

        :param dets: a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        :param image_size: tuple which contains the original image size
        :param motion_model: previously trained motion predictor
        :return: a similar array, where the last column is the object ID.
        """
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # preprocess model input and postprocess output
            # input needs to be normalized and in [cx,cy,w,h] format
            # output needs to be 'unnormalized' and in [x1,y1,x2,y2] format
            model_input = self.trackers[t].get_seq_features()
            model_input = normalize_box(model_input, image_size)
            pos = motion_model(model_input)
            pos = unnormalize_box(pos, image_size)
            pos = box_cxcywh_to_xyxy(pos).squeeze(0)
            pos = np.array(pos)

            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # remove invalid tracks
        for t in reversed(to_del):
            self.trackers.pop(t)

        # match detections and predictions
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = Track(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_last_pos()
            d = box_cxcywh_to_xyxy(d)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
