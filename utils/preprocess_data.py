import os
import os.path as osp

import numpy as np


def preprocess_train(seq_root: str, label_root: str):
    """ changes the ground-truth bbox format from [left,top,w,h] to [cx,cy,w,h]
    also normalizes the box coordinates and splits the ground-truths by tracked objects

    :param seq_root: dataset dir
    :param label_root: output dir for processed ground-truth
    """

    if not osp.exists(label_root):
        os.makedirs(label_root)

    trainer = ["train"]

    # iterate over all given splits
    for tr in trainer:
        seq_root_tr = (osp.join(seq_root, tr))
        info_root_tr = (osp.join(seq_root, 'train'))
        seqs = [s for s in os.listdir(seq_root_tr)]

        # iterate over all sequences in a split
        for seq in seqs:
            print(seq)

            # load sequence infos from seqinfo file
            seq_info = open(osp.join(info_root_tr, seq, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            # load ground-truths from respective ground-truth file
            gt_txt = osp.join(seq_root_tr, seq, 'gt', 'gt.txt')
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            idx = np.lexsort(gt.T[:2, :])
            gt = gt[idx, :]

            seq_label_root = osp.join(label_root, seq, 'img1')
            if not osp.exists(seq_label_root):
                os.makedirs(seq_label_root)

            # iterate over all lines in the ground-truth file for a given sequence
            for fid, tid, x, y, w, h, mark, cls, vis in gt:
                if mark == 0 or not cls == 1:
                    continue
                fid = int(fid)
                tid = int(tid)

                # transform box format from [bb_left, bb_top, bb_width, bb_height] to
                # [bb_center_x, bb_center_y, bb_width, bb_height]
                x += w / 2
                y += h / 2

                # save updated ground-truths in file
                label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(tid))
                # normalize box coordinates (top-left: [0,0], bottom-right: [1,1])
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    fid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)
