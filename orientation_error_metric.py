import keras.backend as K
import numpy as np

def degree_of_separation(oi, oj):
    a = oi - oj
    a = (a + 180) % 360 - 180
    return abs(a)

def mean_orientation_error(class_id_to_azimuth, batch_size=32):

    def metric(gt, pr):
        class_ids = list(class_id_to_azimuth.keys())
        num_azimuths = len(class_ids)

        start_ix = class_ids[0]
        end_ix = class_ids[-1] + 1

        tp = gt * pr
        fn = gt - tp

        si = 0
        for i in range(0, batch_size):
            for j in range(start_ix, end_ix):
                mask_fn = fn[i, :, :, j]
                sj = 0
                ti = K.sum(gt[i, :, :, j])

                for k in range(start_ix, end_ix):
                    if k == j:
                        continue

                    pij = K.sum(mask_fn * pr[i, :, :, k])
                    sj += pij * degree_of_separation(class_id_to_azimuth[j], class_id_to_azimuth[k])

                cond = K.equal(ti, 0)
                val = K.switch(cond, 0.0, sj / ti)
                si += val

        return 1 / num_azimuths * si

    return metric