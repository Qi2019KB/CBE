# -*- coding: utf-8 -*-
from projects.CBE.pose.CBE_DualPose import main as CBE_DualPose

from projects.CBE.classification.CBE_FixMatch import main as CBE_FixMatch



if __name__ == '__main__':
    # CBE for Pose Estimation
    CBE_DualPose('CBE_DualPose', {'dataset': "Mouse", 'train_num': 100, 'num_labeled': 30, 'valid_num': 500})

    # CBE for Classification
    CBE_FixMatch('CBE_FixMatch', {'dataset': "CIFAR10", 'num_labeled': 40})
