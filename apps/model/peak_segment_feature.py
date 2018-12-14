from scipy.stats import *
from scipy.integrate import *
import numpy.linalg as la
import numpy as np


class PeakSegmentFeature:
    __peak_begin = 0.2
    __peak_end = 0.16

    def __init__(self, all_data, sampling_frequency):
        """
        提取峰段特征
        :param all_data: shape=(n,3)
        :param sampling_frequency: 采集频率
        """
        self.all_data = la.norm(all_data, axis=1)
        self.all_peak_feature = np.zeros((0, 5))
        self.peak_cache_data = np.zeros((0, 1))
        # out
        self.peak_feature = np.zeros((0, 1, 5))
        # out
        self.segment_feature = np.zeros((0, 1, 5))
        window_size = 450
        overlap = 0.5
        # 窗口滑动
        for idx in range(0, len(self.all_data) - window_size, int(window_size * overlap)):
            data = self.all_data[idx: idx + window_size]
            data = data.reshape(data.shape[0], 1)
            is_begin = 0
            peak_count = 0
            if len(self.peak_cache_data) != 0:
                is_begin = 1
            before_index = 1
            for j in range(window_size - 10 - 10):
                i = j + 10
                sub_res = np.mean(data[i - 10: i + 10])
                if is_begin == 1:
                    if sub_res < self.__peak_end:
                        is_begin = 0
                        current_frame_window = data[before_index: i, :]
                        if before_index == 1:
                            current_frame_window = np.vstack([self.peak_cache_data, data])
                        frame_size = current_frame_window.shape[0]
                        time = np.linspace(0, frame_size / sampling_frequency - 1 / sampling_frequency,
                                           frame_size)
                        aucval = trapz(time.reshape(time.shape[0], ),
                                       current_frame_window.reshape(current_frame_window.shape[0], ))
                        intensityval = np.mean(current_frame_window)
                        lengthval = frame_size / sampling_frequency
                        kurtosisval = kurtosis(current_frame_window)
                        skewnessval = skew(current_frame_window)
                        peak_feature = np.array([aucval, intensityval, lengthval, kurtosisval[0], skewnessval[0]])
                        peak_feature = peak_feature.reshape(1, peak_feature.shape[0])
                        self.all_peak_feature = np.vstack([self.all_peak_feature, peak_feature])
                        peak_count += 1
                elif is_begin == 0:
                    if sub_res >= self.__peak_begin:
                        is_begin = 1
                        before_index = i

            if is_begin == 1:
                if before_index == 1:
                    self.peak_cache_data = np.vstack([self.peak_cache_data, data])
                else:
                    self.peak_cache_data = data[before_index: window_size, :]
            if peak_count > 0:
                all_peak_size = self.all_peak_feature.shape[0]
                if all_peak_size > 3:
                    peak_aucvar = np.var(self.all_peak_feature[:, 0])
                    peak_intensityvar = np.var(self.all_peak_feature[:, 1])
                    peak_lengthvar = np.var(self.all_peak_feature[:, 2])
                    peak_kurtosisvar = np.var(self.all_peak_feature[:, 3])
                    peak_skewnessvar = np.var(self.all_peak_feature[:, 4])
                    segment_feature_new = np.array(
                        [peak_aucvar, peak_intensityvar, peak_lengthvar, peak_kurtosisvar, peak_skewnessvar])
                    segment_feature_new = segment_feature_new.reshape(1, 1, 5)
                    # out
                    self.segment_feature = np.vstack([self.segment_feature, segment_feature_new])
                if all_peak_size > 5:
                    current_peak = self.all_peak_feature[-5:, :]
                else:
                    current_peak = self.all_peak_feature
                # out
                peak_feature_new = np.array(list(map(lambda x: x / current_peak.shape[0], np.sum(current_peak, 0))))
                peak_feature_new = peak_feature_new.reshape(1, 1, peak_feature_new.shape[0])
                self.peak_feature = np.vstack([self.peak_feature,
                                               peak_feature_new])

    def get_segment_feature(self):
        return self.segment_feature
