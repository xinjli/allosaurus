import numpy as np


def feature_cmvn(feature):

    frame_cnt = feature.shape[0]
    spk_sum = np.sum(feature, axis=0)
    spk_mean = spk_sum / frame_cnt
    spk_square_sum = np.sum(feature*feature, axis=0)
    spk_std = (spk_square_sum / frame_cnt - spk_mean * spk_mean) ** 0.5

    return (feature - spk_mean)/spk_std


def feature_window(feature, window_size=3):

    assert window_size == 3, "only window size 3 is supported"

    feature = np.concatenate((np.roll(feature, 1, axis=0), feature, np.roll(feature, -1, axis=0)), axis=1)
    feature = feature[::3, ]

    return feature

def feature_window_ordered(feature, window_size=3):
    assert window_size == 3, "Window_size must equall 3"

    shape = feature.shape 

    trailing_els = (3-(shape[0] + 1)%3)%3

    windowed = np.full((shape[0] + 1 + trailing_els, shape[1]), feature[-1])
    windowed[0] = feature[0]
    windowed[1:shape[0] + 1] = feature

    windowed.shape = (windowed.size // (shape[1] * 3), shape[1] * 3 )
    return windowed