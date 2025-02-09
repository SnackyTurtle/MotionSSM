import os

import torch
from torch.utils.data import Dataset
import glob
import numpy as np


class MotionDataset(Dataset):
    def __init__(self, seq_len, data_dir):
        self.seq_len = seq_len + 1
        self.data_dir = data_dir
        self.tracks = {}
        self.len_tracks = {}
        self.num_samples = 0

        self.samples_track = {}
        self.cum_ids = {}

        if os.path.isdir(self.data_dir):
            self.seqs = [s for s in os.listdir(self.data_dir)]
            self.seqs.sort()
            last_idx = 0

            # iterate over all sequences in the dataset
            for seq in self.seqs:
                track_paths = os.path.join(self.data_dir + "/" + seq, "img1/*.txt")
                self.tracks[seq] = sorted(glob.glob(track_paths))
                self.len_tracks[seq] = {}

                # iterate over all tracks in a sequence
                for i, pa in enumerate(self.tracks[seq]):
                    len_track = len(np.loadtxt(pa, dtype=np.float32).reshape(-1, 7))
                    self.len_tracks[seq][i] = len_track - self.seq_len
                    self.num_samples += self.len_tracks[seq][i]

                # calculate start ids for each track
                self.samples_track[seq] = [x for x in self.len_tracks[seq].values()]
                self.cum_ids[seq] = [sum(self.samples_track[seq][:i]) + last_idx for i in
                                     range(len(self.samples_track[seq]))]
                last_idx = self.cum_ids[seq][-1] + self.samples_track[seq][-1]

    def __getitem__(self, idx):

        # find corresponding sequence, track and start id for given idx
        for i, seq in enumerate(self.cum_ids):  # 遍历视频序列
            if idx >= self.cum_ids[seq][0]:
                seq_name = seq
                for j, c in enumerate(self.cum_ids[seq]):
                    if idx >= c:
                        idx_track = j
                        idx_start = c
                    else:
                        break
            else:
                break

        # read ground-truth file which contains idx
        track_path = self.tracks[seq_name][idx_track]
        track_gts = np.loadtxt(track_path, dtype=np.float32)

        # extract sequence of boxes from gt
        idx_seq_start = idx - idx_start
        curr_gt = track_gts[idx_seq_start + self.seq_len]
        curr_box = curr_gt[2:6]
        seq_boxes = [track_gts[idx_seq_start + self.seq_len + pos][2:6] for pos in range(self.seq_len)]

        # calculate features from box sequence
        delta_boxes = [seq_boxes[i] - seq_boxes[i-1] for i in range(1,self.seq_len)]
        seq_features = torch.concatenate([torch.tensor(seq_boxes[1:]), torch.tensor(delta_boxes)], dim=1)

        out = {'gt': curr_gt, 'gt_box': curr_box, 'seq_features': seq_features}
        return out


    def __len__(self):
        return self.num_samples

