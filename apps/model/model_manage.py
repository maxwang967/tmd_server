from django.core.cache import cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json

from model.peak_segment_feature import PeakSegmentFeature

use_cuda = False


class TMDNet(nn.Module):
    """
    Transportation Mode Detection Model Based On Deep Learning

    This model was built by PyTorch Framework, needing to be fed with each axis of each type of sensor data
    (lacc_x, lacc_y, lacc_z... for instance). And the output will be the label of specific transportation mode.
    Comments have been carefully noted during the source code, please read carefully to have a better understanding
    of the code.
    I highly recommend you to review the code with PyCharm IDE to enable better experience of testing or refactoring
    process.

    Remember that all tensor shape through the model should follow (N, C, L)
    (N for batch_size, C for channel, L for Length)

    Contact me if further information is required: morningstarwang@outlook.com
    """

    def __init__(self, window_size):
        super(TMDNet, self).__init__()
        if torch.cuda.is_available() and use_cuda:
            print("using GPU")
            # first conv layer
            self.input_conv = nn.Conv1d(1, 64, kernel_size=3).cuda()
            # second conv layer
            self.secondary_conv = nn.Conv1d(64, 128, kernel_size=3).cuda()
            # third conv layer(after concatenate)
            self.third_conv = nn.Conv1d(384, 32, kernel_size=3).cuda()
            # fix pressure conv
            self.fix_conv = nn.Conv1d(128, 128, kernel_size=3).cuda()
            # pooling layer
            self.pooling = nn.MaxPool1d(2, stride=2).cuda()
            # lstm layer
            # param: input_dim, hidden_layer_dim, the number of lstm layers
            self.lstm = nn.LSTM(229, 128, num_layers=window_size).cuda()
            # first full connection layer
            self.first_fc = nn.Linear(128, 128).cuda()
            # second full connection layer
            self.second_fc = nn.Linear(128, 256).cuda()
            # third full connection layer
            self.third_fc = nn.Linear(256, 512).cuda()
            # fourth full connection layer
            self.fourth_fc = nn.Linear(512, 1024).cuda()
            # output full connection layer
            self.output_fc = nn.Linear(1024, 6).cuda()
        else:
            print("using CPU")
            # first conv layer
            self.input_conv = nn.Conv1d(1, 64, kernel_size=3)
            # second conv layer
            self.secondary_conv = nn.Conv1d(64, 128, kernel_size=3)
            # third conv layer(after concatenate)
            self.third_conv = nn.Conv1d(384, 32, kernel_size=3)
            # fix pressure conv
            self.fix_conv = nn.Conv1d(128, 128, kernel_size=3)
            # pooling layer
            self.pooling = nn.MaxPool1d(2, stride=2)
            # lstm layer
            # param: input_dim, hidden_layer_dim, the number of lstm layers
            self.lstm = nn.LSTM(229, 128, num_layers=4)
            # first full connection layer
            self.first_fc = nn.Linear(128, 128)
            # second full connection layer
            self.second_fc = nn.Linear(128, 256)
            # third full connection layer
            self.third_fc = nn.Linear(256, 512)
            # fourth full connection layer
            self.fourth_fc = nn.Linear(512, 1024)
            # output full connection layer
            self.output_fc = nn.Linear(1024, 6)

    def forward(self, x_data):
        if torch.cuda.is_available() and use_cuda:
            # region original data, shape=(N, 13, window_size)
            lacc_x = x_data[:, 0:1, :].cuda()
            lacc_y = x_data[:, 1:2, :].cuda()
            lacc_z = x_data[:, 2:3, :].cuda()

            gyr_x = x_data[:, 3:4, :].cuda()
            gyr_y = x_data[:, 4:5, :].cuda()
            gyr_z = x_data[:, 5:6, :].cuda()

            mag_x = x_data[:, 6:7, :].cuda()
            mag_y = x_data[:, 7:8, :].cuda()
            mag_z = x_data[:, 8:9, :].cuda()

            pressure = x_data[:, 9:10, :].cuda()
            # endregion
        else:
            # region original data, shape=(N, 13, window_size)
            lacc_x = x_data[:, 0:1, :]
            lacc_y = x_data[:, 1:2, :]
            lacc_z = x_data[:, 2:3, :]

            gyr_x = x_data[:, 3:4, :]
            gyr_y = x_data[:, 4:5, :]
            gyr_z = x_data[:, 5:6, :]

            mag_x = x_data[:, 6:7, :]
            mag_y = x_data[:, 7:8, :]
            mag_z = x_data[:, 8:9, :]

            pressure = x_data[:, 9:10, :]
            # endregion

        # region lacc, output=(N, 32, 54)
        # region lacc_x, output.shape=(N, 128, 111)
        layer_lacc_x = self.input_conv(lacc_x)
        layer_lacc_x = F.relu(layer_lacc_x)
        # now: (N, 64, 448)
        layer_lacc_x = self.pooling(layer_lacc_x)
        # now: (N, 64, 224)
        layer_lacc_x = self.secondary_conv(layer_lacc_x)
        layer_lacc_x = F.relu(layer_lacc_x)
        # now: (N, 128, 222)
        layer_lacc_x = self.pooling(layer_lacc_x)
        # now: (N, 128, 111)
        # endregion
        # region lacc_y, output.shape=(N, 128, 111)
        layer_lacc_y = self.input_conv(lacc_y)
        layer_lacc_y = F.relu(layer_lacc_y)
        # now: (N, 64, 448)
        layer_lacc_y = self.pooling(layer_lacc_y)
        # now: (N, 64, 224)
        layer_lacc_y = self.secondary_conv(layer_lacc_y)
        layer_lacc_y = F.relu(layer_lacc_y)
        # now: (N, 128, 222)
        layer_lacc_y = self.pooling(layer_lacc_y)
        # now: (N, 128, 111)
        # endregion
        # region lacc_z, output.shape=(N, 128, 111)
        layer_lacc_z = self.input_conv(lacc_z)
        layer_lacc_z = F.relu(layer_lacc_z)
        # now: (N, 64, 448)
        layer_lacc_z = self.pooling(layer_lacc_z)
        # now: (N, 64, 224)
        layer_lacc_z = self.secondary_conv(layer_lacc_z)
        layer_lacc_z = F.relu(layer_lacc_z)
        # now: (N, 128, 222)
        layer_lacc_z = self.pooling(layer_lacc_z)
        # now: (N, 128, 111)
        # endregion
        layer_lacc = torch.cat(
            (layer_lacc_x,
             layer_lacc_y,
             layer_lacc_z),
            dim=1
        )
        layer_lacc = self.third_conv(layer_lacc)
        layer_lacc = F.relu(layer_lacc)
        layer_lacc = self.pooling(layer_lacc)
        # endregion lacc

        # region gyr, output=(N, 32, 54)
        # region gyr_x, output.shape=(N, 128, 111)
        layer_gyr_x = self.input_conv(gyr_x)
        layer_gyr_x = F.relu(layer_gyr_x)
        # now: (N, 64, 448)
        layer_gyr_x = self.pooling(layer_gyr_x)
        # now: (N, 64, 224)
        layer_gyr_x = self.secondary_conv(layer_gyr_x)
        layer_gyr_x = F.relu(layer_gyr_x)
        # now: (N, 128, 222)
        layer_gyr_x = self.pooling(layer_gyr_x)
        # now: (N, 128, 111)
        # endregion
        # region gyr_y, output.shape=(N, 128, 111)
        layer_gyr_y = self.input_conv(gyr_y)
        layer_gyr_y = F.relu(layer_gyr_y)
        # now: (N, 64, 448)
        layer_gyr_y = self.pooling(layer_gyr_y)
        # now: (N, 64, 224)
        layer_gyr_y = self.secondary_conv(layer_gyr_y)
        layer_gyr_y = F.relu(layer_gyr_y)
        # now: (N, 128, 222)
        layer_gyr_y = self.pooling(layer_gyr_y)
        # now: (N, 128, 111)
        # endregion
        # region gyr_z, output.shape=(N, 128, 111)
        layer_gyr_z = self.input_conv(gyr_z)
        layer_gyr_z = F.relu(layer_gyr_z)
        # now: (N, 64, 448)
        layer_gyr_z = self.pooling(layer_gyr_z)
        # now: (N, 64, 224)
        layer_gyr_z = self.secondary_conv(layer_gyr_z)
        layer_gyr_z = F.relu(layer_gyr_z)
        # now: (N, 128, 222)
        layer_gyr_z = self.pooling(layer_gyr_z)
        # now: (N, 128, 111)
        # endregion
        layer_gyr = torch.cat(
            (layer_gyr_x,
             layer_gyr_y,
             layer_gyr_z),
            dim=1
        )
        layer_gyr = self.third_conv(layer_gyr)
        layer_gyr = F.relu(layer_gyr)
        layer_gyr = self.pooling(layer_gyr)
        # endregion gyr

        # region mag, output=(N, 32, 54)
        # region mag_x, output.shape=(N, 128, 111)
        layer_mag_x = self.input_conv(mag_x)
        layer_mag_x = F.relu(layer_mag_x)
        # now: (N, 64, 448)
        layer_mag_x = self.pooling(layer_mag_x)
        # now: (N, 64, 224)
        layer_mag_x = self.secondary_conv(layer_mag_x)
        layer_mag_x = F.relu(layer_mag_x)
        # now: (N, 128, 222)
        layer_mag_x = self.pooling(layer_mag_x)
        # now: (N, 128, 111)
        # endregion
        # region mag_y, output.shape=(N, 128, 111)
        layer_mag_y = self.input_conv(mag_y)
        layer_mag_y = F.relu(layer_mag_y)
        # now: (N, 64, 448)
        layer_mag_y = self.pooling(layer_mag_y)
        # now: (N, 64, 224)
        layer_mag_y = self.secondary_conv(layer_mag_y)
        layer_mag_y = F.relu(layer_mag_y)
        # now: (N, 128, 222)
        layer_mag_y = self.pooling(layer_mag_y)
        # now: (N, 128, 111)
        # endregion
        # region mag_z, output.shape=(N, 128, 111)
        layer_mag_z = self.input_conv(mag_z)
        layer_mag_z = F.relu(layer_mag_z)
        # now: (N, 64, 448)
        layer_mag_z = self.pooling(layer_mag_z)
        # now: (N, 64, 224)
        layer_mag_z = self.secondary_conv(layer_mag_z)
        layer_mag_z = F.relu(layer_mag_z)
        # now: (N, 128, 222)
        layer_mag_z = self.pooling(layer_mag_z)
        # now: (N, 128, 111)
        # endregion
        layer_mag = torch.cat(
            (layer_mag_x,
             layer_mag_y,
             layer_mag_z),
            dim=1
        )
        layer_mag = self.third_conv(layer_mag)
        layer_mag = F.relu(layer_mag)
        layer_mag = self.pooling(layer_mag)
        # endregion mag

        # region pressure, output=(N, 128, 54)
        layer_pressure = self.input_conv(pressure)
        layer_pressure = F.relu(layer_pressure)
        layer_pressure = self.pooling(layer_pressure)
        # now: (N, 64, 224)
        layer_pressure = self.secondary_conv(layer_pressure)
        layer_pressure = F.relu(layer_pressure)
        layer_pressure = self.pooling(layer_pressure)
        # now: (N, 128, 111)
        layer_pressure = self.fix_conv(layer_pressure)
        layer_pressure = F.relu(layer_pressure)
        layer_pressure = self.pooling(layer_pressure)
        # now: (N, 128, 54)
        # endregion

        # region segment_feature, output=(N, 5, 54)
        # number to measure how many windows are required to calculate segment feature
        k = 16
        presegment_lacc_data = np.concatenate((lacc_x, lacc_y, lacc_z), axis=1)
        # now: (N, 3, 450)
        presegment_lacc_data_list = []
        # output data
        layer_segment = np.empty((0, 5, 54))
        batches = int(len(presegment_lacc_data) / k)
        # divided by 16(1.2min) to calculate segment feature
        for i in range(batches):
            presegment_lacc_data_list.append(presegment_lacc_data[i * k: (i + 1) * k, :, :])
        # foreach in the list
        for block in presegment_lacc_data_list:
            # block of output data, output=(16, 5, 54)
            layer_segment_block = np.empty((0, 5, 54))
            # block.shape=(16, 3, 450)
            block = np.swapaxes(block, 1, 2)
            block = np.reshape(block, (k * 450, 3))
            peak_segment_feature = PeakSegmentFeature(all_data=block, sampling_frequency=100)
            # segment_feature.shape=(?, 1, 5)
            segment_feature = peak_segment_feature.get_segment_feature()
            # not enough, fill with zero
            if len(segment_feature) <= k:
                for i in range(len(segment_feature)):
                    output_feature = np.empty((1, 5, 0))
                    feature = segment_feature[i]
                    feature = np.reshape(feature, (5,))
                    feature = np.reshape(feature, (5, 1))
                    feature = np.reshape(feature, (1, 5, 1))
                    for j in range(54):
                        output_feature = np.concatenate([output_feature, feature], axis=2)
                    layer_segment_block = np.vstack([layer_segment_block, output_feature])
                fill = np.zeros((1, 5, 54))
                for i in range(k - len(layer_segment_block)):
                    layer_segment_block = np.vstack([layer_segment_block, fill])
            else:
                for i in range(k):
                    output_feature = np.empty((1, 5, 0))
                    feature = segment_feature[i]
                    feature = np.reshape(feature, (5,))
                    feature = np.reshape(feature, (5, 1))
                    feature = np.reshape(feature, (1, 5, 1))
                    for j in range(54):
                        output_feature = np.concatenate([output_feature, feature], axis=2)
                    layer_segment_block = np.vstack([layer_segment_block, output_feature])
            # stack the block to the output
            layer_segment = np.vstack([layer_segment, layer_segment_block])
        layer_segment = torch.Tensor(layer_segment)
        # endregion

        # region CNN, output=(N, 229, 54)
        layer_cnn = torch.cat(
            (layer_lacc, layer_gyr, layer_mag, layer_pressure, layer_segment),
            dim=1
        )
        # endregion
        # region LSTM, output=(N, 128)
        # old input, shape=(0 batch_size, 1 input_dim, 2 seq_length)
        # what lstm need is: shape=(2 seq_length, 0 batch_size, 1 input_dim)
        lstm_input = layer_cnn.permute(2, 0, 1)
        layer_lstm, _ = self.lstm(lstm_input)
        layer_lstm = layer_lstm[-1, :, :]
        layer_lstm = torch.tanh(layer_lstm)
        # now: (N, 128)
        # endregion
        # region FC, output
        # layer 1
        layer_fc = self.first_fc(layer_lstm)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # layer 2
        layer_fc = self.second_fc(layer_fc)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # layer 3
        layer_fc = self.third_fc(layer_fc)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # layer 4
        layer_fc = self.fourth_fc(layer_fc)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # now: (N, 1024)
        # layer output
        layer_fc = self.output_fc(layer_fc)
        # layer_fc = F.log_softmax(layer_fc, dim=1)
        # if torch.cuda.is_available():
        #     print("on gpu")
        #     self.cuda()
        return layer_fc


class TMDNetWithoutSegment(nn.Module):
    """
    Transportation Mode Detection Model Based On Deep Learning

    This model was built by PyTorch Framework, needing to be fed with each axis of each type of sensor data
    (lacc_x, lacc_y, lacc_z... for instance). And the output will be the label of specific transportation mode.
    Comments have been carefully noted during the source code, please read carefully to have a better understanding
    of the code.
    I highly recommend you to review the code with PyCharm IDE to enable better experience of testing or refactoring
    process.

    Remember that all tensor shape through the model should follow (N, C, L)
    (N for batch_size, C for channel, L for Length)

    Contact me if further information is required: morningstarwang@outlook.com
    """

    def __init__(self, window_size):
        super(TMDNetWithoutSegment, self).__init__()
        if torch.cuda.is_available() and use_cuda:
            print("using GPU")
            # first conv layer
            self.input_conv = nn.Conv1d(1, 64, kernel_size=3).cuda()
            # second conv layer
            self.secondary_conv = nn.Conv1d(64, 128, kernel_size=3).cuda()
            # third conv layer(after concatenate)
            self.third_conv = nn.Conv1d(384, 32, kernel_size=3).cuda()
            # fix pressure conv
            self.fix_conv = nn.Conv1d(128, 128, kernel_size=3).cuda()
            # pooling layer
            self.pooling = nn.MaxPool1d(2, stride=2).cuda()
            # lstm layer
            # param: input_dim, hidden_layer_dim, the number of lstm layers
            self.lstm = nn.LSTM(224, 128, num_layers=window_size).cuda()
            # first full connection layer
            self.first_fc = nn.Linear(128, 128).cuda()
            # second full connection layer
            self.second_fc = nn.Linear(128, 256).cuda()
            # third full connection layer
            self.third_fc = nn.Linear(256, 512).cuda()
            # fourth full connection layer
            self.fourth_fc = nn.Linear(512, 1024).cuda()
            # output full connection layer
            self.output_fc = nn.Linear(1024, 6).cuda()
        else:
            print("using CPU")
            # first conv layer
            self.input_conv = nn.Conv1d(1, 64, kernel_size=3)
            # second conv layer
            self.secondary_conv = nn.Conv1d(64, 128, kernel_size=3)
            # third conv layer(after concatenate)
            self.third_conv = nn.Conv1d(384, 32, kernel_size=3)
            # fix pressure conv
            self.fix_conv = nn.Conv1d(128, 128, kernel_size=3)
            # pooling layer
            self.pooling = nn.MaxPool1d(2, stride=2)
            # lstm layer
            # param: input_dim, hidden_layer_dim, the number of lstm layers
            self.lstm = nn.LSTM(224, 128, num_layers=4)
            # first full connection layer
            self.first_fc = nn.Linear(128, 128)
            # second full connection layer
            self.second_fc = nn.Linear(128, 256)
            # third full connection layer
            self.third_fc = nn.Linear(256, 512)
            # fourth full connection layer
            self.fourth_fc = nn.Linear(512, 1024)
            # output full connection layer
            self.output_fc = nn.Linear(1024, 6)

    def forward(self, x_data):
        if torch.cuda.is_available() and use_cuda:
            # region original data, shape=(N, 13, window_size)
            lacc_x = x_data[:, 0:1, :].cuda()
            lacc_y = x_data[:, 1:2, :].cuda()
            lacc_z = x_data[:, 2:3, :].cuda()

            gyr_x = x_data[:, 3:4, :].cuda()
            gyr_y = x_data[:, 4:5, :].cuda()
            gyr_z = x_data[:, 5:6, :].cuda()

            mag_x = x_data[:, 6:7, :].cuda()
            mag_y = x_data[:, 7:8, :].cuda()
            mag_z = x_data[:, 8:9, :].cuda()

            pressure = x_data[:, 9:10, :].cuda()
            # endregion
        else:
            # region original data, shape=(N, 13, window_size)
            lacc_x = x_data[:, 0:1, :]
            lacc_y = x_data[:, 1:2, :]
            lacc_z = x_data[:, 2:3, :]

            gyr_x = x_data[:, 3:4, :]
            gyr_y = x_data[:, 4:5, :]
            gyr_z = x_data[:, 5:6, :]

            mag_x = x_data[:, 6:7, :]
            mag_y = x_data[:, 7:8, :]
            mag_z = x_data[:, 8:9, :]

            pressure = x_data[:, 9:10, :]
            # endregion

        # region lacc, output=(N, 32, 54)
        # region lacc_x, output.shape=(N, 128, 111)
        layer_lacc_x = self.input_conv(lacc_x)
        layer_lacc_x = F.relu(layer_lacc_x)
        # now: (N, 64, 448)
        layer_lacc_x = self.pooling(layer_lacc_x)
        # now: (N, 64, 224)
        layer_lacc_x = self.secondary_conv(layer_lacc_x)
        layer_lacc_x = F.relu(layer_lacc_x)
        # now: (N, 128, 222)
        layer_lacc_x = self.pooling(layer_lacc_x)
        # now: (N, 128, 111)
        # endregion
        # region lacc_y, output.shape=(N, 128, 111)
        layer_lacc_y = self.input_conv(lacc_y)
        layer_lacc_y = F.relu(layer_lacc_y)
        # now: (N, 64, 448)
        layer_lacc_y = self.pooling(layer_lacc_y)
        # now: (N, 64, 224)
        layer_lacc_y = self.secondary_conv(layer_lacc_y)
        layer_lacc_y = F.relu(layer_lacc_y)
        # now: (N, 128, 222)
        layer_lacc_y = self.pooling(layer_lacc_y)
        # now: (N, 128, 111)
        # endregion
        # region lacc_z, output.shape=(N, 128, 111)
        layer_lacc_z = self.input_conv(lacc_z)
        layer_lacc_z = F.relu(layer_lacc_z)
        # now: (N, 64, 448)
        layer_lacc_z = self.pooling(layer_lacc_z)
        # now: (N, 64, 224)
        layer_lacc_z = self.secondary_conv(layer_lacc_z)
        layer_lacc_z = F.relu(layer_lacc_z)
        # now: (N, 128, 222)
        layer_lacc_z = self.pooling(layer_lacc_z)
        # now: (N, 128, 111)
        # endregion
        layer_lacc = torch.cat(
            (layer_lacc_x,
             layer_lacc_y,
             layer_lacc_z),
            dim=1
        )
        layer_lacc = self.third_conv(layer_lacc)
        layer_lacc = F.relu(layer_lacc)
        layer_lacc = self.pooling(layer_lacc)
        # endregion lacc

        # region gyr, output=(N, 32, 54)
        # region gyr_x, output.shape=(N, 128, 111)
        layer_gyr_x = self.input_conv(gyr_x)
        layer_gyr_x = F.relu(layer_gyr_x)
        # now: (N, 64, 448)
        layer_gyr_x = self.pooling(layer_gyr_x)
        # now: (N, 64, 224)
        layer_gyr_x = self.secondary_conv(layer_gyr_x)
        layer_gyr_x = F.relu(layer_gyr_x)
        # now: (N, 128, 222)
        layer_gyr_x = self.pooling(layer_gyr_x)
        # now: (N, 128, 111)
        # endregion
        # region gyr_y, output.shape=(N, 128, 111)
        layer_gyr_y = self.input_conv(gyr_y)
        layer_gyr_y = F.relu(layer_gyr_y)
        # now: (N, 64, 448)
        layer_gyr_y = self.pooling(layer_gyr_y)
        # now: (N, 64, 224)
        layer_gyr_y = self.secondary_conv(layer_gyr_y)
        layer_gyr_y = F.relu(layer_gyr_y)
        # now: (N, 128, 222)
        layer_gyr_y = self.pooling(layer_gyr_y)
        # now: (N, 128, 111)
        # endregion
        # region gyr_z, output.shape=(N, 128, 111)
        layer_gyr_z = self.input_conv(gyr_z)
        layer_gyr_z = F.relu(layer_gyr_z)
        # now: (N, 64, 448)
        layer_gyr_z = self.pooling(layer_gyr_z)
        # now: (N, 64, 224)
        layer_gyr_z = self.secondary_conv(layer_gyr_z)
        layer_gyr_z = F.relu(layer_gyr_z)
        # now: (N, 128, 222)
        layer_gyr_z = self.pooling(layer_gyr_z)
        # now: (N, 128, 111)
        # endregion
        layer_gyr = torch.cat(
            (layer_gyr_x,
             layer_gyr_y,
             layer_gyr_z),
            dim=1
        )
        layer_gyr = self.third_conv(layer_gyr)
        layer_gyr = F.relu(layer_gyr)
        layer_gyr = self.pooling(layer_gyr)
        # endregion gyr

        # region mag, output=(N, 32, 54)
        # region mag_x, output.shape=(N, 128, 111)
        layer_mag_x = self.input_conv(mag_x)
        layer_mag_x = F.relu(layer_mag_x)
        # now: (N, 64, 448)
        layer_mag_x = self.pooling(layer_mag_x)
        # now: (N, 64, 224)
        layer_mag_x = self.secondary_conv(layer_mag_x)
        layer_mag_x = F.relu(layer_mag_x)
        # now: (N, 128, 222)
        layer_mag_x = self.pooling(layer_mag_x)
        # now: (N, 128, 111)
        # endregion
        # region mag_y, output.shape=(N, 128, 111)
        layer_mag_y = self.input_conv(mag_y)
        layer_mag_y = F.relu(layer_mag_y)
        # now: (N, 64, 448)
        layer_mag_y = self.pooling(layer_mag_y)
        # now: (N, 64, 224)
        layer_mag_y = self.secondary_conv(layer_mag_y)
        layer_mag_y = F.relu(layer_mag_y)
        # now: (N, 128, 222)
        layer_mag_y = self.pooling(layer_mag_y)
        # now: (N, 128, 111)
        # endregion
        # region mag_z, output.shape=(N, 128, 111)
        layer_mag_z = self.input_conv(mag_z)
        layer_mag_z = F.relu(layer_mag_z)
        # now: (N, 64, 448)
        layer_mag_z = self.pooling(layer_mag_z)
        # now: (N, 64, 224)
        layer_mag_z = self.secondary_conv(layer_mag_z)
        layer_mag_z = F.relu(layer_mag_z)
        # now: (N, 128, 222)
        layer_mag_z = self.pooling(layer_mag_z)
        # now: (N, 128, 111)
        # endregion
        layer_mag = torch.cat(
            (layer_mag_x,
             layer_mag_y,
             layer_mag_z),
            dim=1
        )
        layer_mag = self.third_conv(layer_mag)
        layer_mag = F.relu(layer_mag)
        layer_mag = self.pooling(layer_mag)
        # endregion mag

        # region pressure, output=(N, 128, 54)
        layer_pressure = self.input_conv(pressure)
        layer_pressure = F.relu(layer_pressure)
        layer_pressure = self.pooling(layer_pressure)
        # now: (N, 64, 224)
        layer_pressure = self.secondary_conv(layer_pressure)
        layer_pressure = F.relu(layer_pressure)
        layer_pressure = self.pooling(layer_pressure)
        # now: (N, 128, 111)
        layer_pressure = self.fix_conv(layer_pressure)
        layer_pressure = F.relu(layer_pressure)
        layer_pressure = self.pooling(layer_pressure)
        # now: (N, 128, 54)
        # endregion

        # region CNN, output=(N, 224, 54)
        layer_cnn = torch.cat(
            (layer_lacc, layer_gyr, layer_mag, layer_pressure),
            dim=1
        )
        # endregion
        # region LSTM, output=(N, 128)
        # old input, shape=(0 batch_size, 1 input_dim, 2 seq_length)
        # what lstm need is: shape=(2 seq_length, 0 batch_size, 1 input_dim)
        lstm_input = layer_cnn.permute(2, 0, 1)
        layer_lstm, _ = self.lstm(lstm_input)
        layer_lstm = layer_lstm[-1, :, :]
        layer_lstm = torch.tanh(layer_lstm)
        # now: (N, 128)
        # endregion
        # region FC, output
        # layer 1
        layer_fc = self.first_fc(layer_lstm)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # layer 2
        layer_fc = self.second_fc(layer_fc)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # layer 3
        layer_fc = self.third_fc(layer_fc)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # layer 4
        layer_fc = self.fourth_fc(layer_fc)
        layer_fc = F.relu(layer_fc)
        layer_fc = F.dropout(layer_fc, 0.2)
        # now: (N, 1024)
        # layer output
        layer_fc = self.output_fc(layer_fc)
        # layer_fc = F.log_softmax(layer_fc, dim=1)
        # if torch.cuda.is_available():
        #     print("on gpu")
        #     self.cuda()
        return layer_fc


def __feature_normalize(dataset, mu, sigma):
    # mu = np.mean(dataset, axis=1)
    # sigma = np.std(dataset, axis=1)
    return (dataset - mu) / sigma


def __initialize_model():
    if torch.cuda.is_available() and use_cuda:
        model = TMDNet(450).cuda()
        model_default = TMDNetWithoutSegment(450).cuda()
    else:
        model = TMDNet(450)
        model_default = TMDNetWithoutSegment(450)
    model.load_state_dict(torch.load("weights_of_model.bin")["state_dict"])
    model_default.load_state_dict(torch.load("weights_of_model_default.bin")["state_dict"])
    cache.set("model_default", model_default, None)
    return model


def __get_model_with_username(username):
    print("username=%s" % username)
    # indicate if the model is under training
    training_flag = cache.get("model_training_flag_%s" % username)
    model = None
    # if not training new model, use model
    if training_flag is None or not training_flag:
        model = cache.get("model_%s" % username)
    # if training new model, use the backup of model instead
    elif training_flag:
        model = cache.get("model_bak_%s" % username)
    # if first time or server has been shutdown, cache a new model
    if model is None:
        model = __initialize_model()
        cache.set("model_%s" % username, model, None)
        cache.set("model_bak_%s" % username, model, None)
        cache.set("model_training_flag_%s" % username, False, None)
    return model


def __push_data_with_username(username, window_data):
    print("username=%s" % username)
    print("window_data.shape=%s" % str(window_data.shape))
    all_data = cache.get("data_%s" % username)
    if all_data is not None:
        print("all_data_cache.shape=%s" % str(all_data.shape))
    if all_data is None:
        all_data = np.empty((0, 10, 450))
        all_data = np.vstack([all_data, window_data])
        cache.set("data_%s" % username, all_data, None)
        return all_data
    if all_data.shape[0] == 16:
        all_data = all_data[1:, :, :]
    all_data = np.vstack([all_data, window_data])
    print("all_data_now.shape=%s" % str(all_data.shape))
    cache.set("data_%s" % username, all_data, None)
    if all_data.shape[0] < 16:
        return all_data
    return all_data


def __train_model_with_username(username, all_data, label):
    print("username=%s" % username)
    cache.set("model_training_flag_%s" % username, True, None)
    # judge if the model with this username exists
    model = cache.get("model_%s" % username)
    # no model currently exists
    if model is None:
        cache.set("model_training_flag_%s" % username, False, None)
        return False
    # start training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    output = model(all_data)
    prediction = output.data.max(1)[1]
    tmp = prediction.eq(label)
    accuracy = (tmp.float().sum() / 1) * 100
    label = np.array([
        label, label, label, label, label, label, label, label, label, label, label, label, label, label, label, label
    ])
    label = torch.Tensor(label)
    label = label.long()
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    torch.save({
        'epoch': 1,
        'state_dict': model.state_dict(),
        'best_acc': accuracy,
        'optimizer': optimizer.state_dict()
    }, "checkpoint_%s.bin" % username)
    cache.set("model_%s" % username, model, None)
    cache.set("model_bak_%s" % username, model, None)
    # end training
    cache.set("model_training_flag_%s" % username, False, None)
    return True


def __resolve_request_data(request_data):
    lacc_list = request_data["mLAccList"]
    gyr_list = request_data["mGyrList"]
    mag_list = request_data["mMagList"]
    pressure_list = request_data["mPressureList"]
    label = request_data["label"]
    laccx = []
    laccy = []
    laccz = []
    for lacc in lacc_list:
        laccx.append(lacc["x"])
        laccy.append(lacc["y"])
        laccz.append(lacc["z"])
    laccx = np.array(laccx, dtype=float)
    laccy = np.array(laccy, dtype=float)
    laccz = np.array(laccz, dtype=float)
    laccx = laccx.reshape((450, 1))
    laccx = laccx[np.newaxis, :, :]
    laccx = __feature_normalize(laccx, 0.020449250321555236, 1.4135227799134187)
    laccy = laccy.reshape((450, 1))
    laccy = laccy[np.newaxis, :, :]
    laccy = __feature_normalize(laccy, -0.21382969391583076, 1.4657551566826752)
    laccz = laccz.reshape((450, 1))
    laccz = laccz[np.newaxis, :, :]
    laccz = __feature_normalize(laccz, -0.042455746000880486, 1.6746452878336602)

    gyrx = []
    gyry = []
    gyrz = []
    for gyr in gyr_list:
        gyrx.append(gyr["x"])
        gyry.append(gyr["y"])
        gyrz.append(gyr["z"])
    gyrx = np.array(gyrx, dtype=float)
    gyry = np.array(gyry, dtype=float)
    gyrz = np.array(gyrz, dtype=float)
    gyrx = gyrx.reshape((450, 1))
    gyrx = gyrx[np.newaxis, :, :]
    gyrx = __feature_normalize(gyrx, 0.00010329724720063263, 0.42976102252834586)
    gyry = gyry.reshape((450, 1))
    gyry = gyry[np.newaxis, :, :]
    gyry = __feature_normalize(gyry, -3.21561107038865e-05, 0.43243989937318317)
    gyrz = gyrz.reshape((450, 1))
    gyrz = gyrz[np.newaxis, :, :]
    gyrz = __feature_normalize(gyrz, -0.0003747888249400768, 0.6099038632196319)

    magx = []
    magy = []
    magz = []
    for mag in mag_list:
        magx.append(mag["x"])
        magy.append(mag["y"])
        magz.append(mag["z"])
    magx = np.array(magx, dtype=float)
    magy = np.array(magy, dtype=float)
    magz = np.array(magz, dtype=float)
    magx = magx.reshape((450, 1))
    magx = magx[np.newaxis, :, :]
    magx = __feature_normalize(magx, -4.109216302434363, 30.1110591791075)
    magy = magy.reshape((450, 1))
    magy = magy[np.newaxis, :, :]
    magy = __feature_normalize(magy, -13.798764868705199, 37.25622430457126)
    magz = magz.reshape((450, 1))
    magz = magz[np.newaxis, :, :]
    magz = __feature_normalize(magz, -23.71686793426817, 44.31135114875818)

    pressure_list = np.array(pressure_list)
    pressure_list = pressure_list.reshape((450, 1))
    pressure_list = pressure_list[np.newaxis, :, :]
    pressure_list = __feature_normalize(pressure_list, 632994.7614423251, 10249435.389978947)

    window_data = np.concatenate([laccx, laccy, laccz, gyrx, gyry, gyrz, magx, magy, magz, pressure_list], axis=2)
    window_data = np.swapaxes(window_data, 1, 2)
    print("request_data_window_size=%s" % str(window_data.shape))
    return window_data, int(label)


def get_result_with_username(username, request_data):
    window_data, label = __resolve_request_data(request_data)
    # get first or old model with username
    model = __get_model_with_username(username)
    # push new data to queue
    all_data = __push_data_with_username(username, window_data)
    if all_data.shape[0] < 16:
        print("using model without segment")
        model_default = cache.get("model_default")
        model_default.eval()
        print(all_data.shape)
        all_data = torch.Tensor(all_data)
        output = model_default(all_data)
        prediction = output.data.max(1)[1]
        print("prediction=%s" % str(prediction))
        return prediction
    print("all_data.shape=%s" % str(all_data.shape))
    # get result from this first or old model
    model.eval()
    all_data = torch.Tensor(all_data)
    output = model(all_data)
    prediction = output.data.max(1)[1]
    print("prediction=%s" % str(prediction))
    # training new model
    train_success = __train_model_with_username(username, all_data, label)
    if train_success:
        print("train success")
    else:
        print("train failed")
    return prediction


