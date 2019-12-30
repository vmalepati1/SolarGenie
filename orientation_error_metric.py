import keras.backend as K
import numpy as np

def mean_orientation_error(azimuth_angle_dict):

    def metric(gt, pr):
        # Time: O(B * C * (C - 1))
        batch_size = gt.shape[0]
        num_azimuths = gt.shape[3]

        for i in range(0, batch_size):
            for j in range(0, num_azimuths):
                mask_gt = gt[i, :, :, j]
                mask_pr = pr[i, :, :, j]

                # True positives
                tp = mask_gt * mask_pr
                fn = mask_gt - tp

                for k in range(0, num_azimuths):
                    if k == j:
                        continue

                    mask_pot_false_pred = pr[i, :, :, j]

                    pij = K.sum(mask_pot_false_pred * fn, axis=1)


    return metric

x = K.constant(np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]))

print(K.sum(x, axis=2))