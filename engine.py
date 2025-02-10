from collections.abc import Iterable
from tracker.sort import Sort

import torch
import os
import glob
import numpy as np
import time


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, writer, epoch: int):
    model.train()
    criterion.train()

    optimizer.zero_grad()
    for i, batch in enumerate(data_loader):

        features = batch['seq_features'].to(device)
        gt_box = batch['gt_box'].to(device)

        pred_box = model(features)

        loss = criterion(pred_box, gt_box)
        loss.backward()

        # gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('train_loss', loss, epoch * len(data_loader) + i)


def track(model: torch.nn.Module, phase: str, seq_path: str, split: str, output_dir: str):

    total_time = 0.0
    total_frames = 0

    # find all sequence directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pattern = os.path.join(seq_path, phase, '*')
    seq_file_names = glob.glob(pattern)
    seq_file_names.sort()

    # iterate over all sequences in dataset
    for seq_dets_fn in seq_file_names:
        mot_tracker = Sort(max_age=1,
                           min_hits=3,
                           iou_threshold=0.3)

        # load detections
        seq_dets = np.loadtxt(seq_dets_fn + '/det.txt', delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        # extract image sizes from corresponding info files
        # needed as the motion model is trained on normalized coordinates
        import configparser
        config = configparser.ConfigParser()
        config.read(seq_dets_fn.replace(phase, split) + '/seqinfo.ini')
        image_size = (int(config.get('Sequence', 'imWidth')), int(config.get('Sequence', 'imHeight')))

        # iterate over all frames in the sequence
        with open(os.path.join(output_dir, '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 1:]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                start_time = time.time()
                with torch.no_grad():
                    trackers = mot_tracker.update(dets, image_size, model)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                # save ids and boxes
                # boxes are saved as left, top, width, height
                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
    total_time, total_frames, total_frames / total_time))



