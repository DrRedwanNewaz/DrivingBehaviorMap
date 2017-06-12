import numpy as np


class data_base(object):
    def __init__(self, id_index, window_size=15):
        self.num_samples = np.shape(id_index)[0]
        self.id_index = id_index
        self.data_set_windowed = []
        self.window_size = window_size

    def window(self, fseq, window_size=15):
        for i in range(len(fseq) - window_size + 1):
            yield fseq[i:i + window_size]

    def concat_drive(self):
        behavior_dataset = np.array(self.data_set_windowed[0])
        for i in range(self.num_samples - 1):
            behavior_dataset = np.vstack((self.data_set_windowed[i + 1], behavior_dataset))
        return behavior_dataset

    def read_files(self):
        # read raw data
        for driver_id in (self.id_index):
            X_windowed = []
            filename = 'dataset/driver_%d.txt' % driver_id
            raw_file = np.loadtxt(filename)
            for j in self.window(raw_file, self.window_size):
                X_windowed.append(j)
            X_windowed = np.array(X_windowed)
            # X_windowed = X_windowed.reshape((X_windowed.shape[0], -1), order='F')
        #     self.data_set_windowed.append(X_windowed)
        # behavior_dataset = self.concat_drive()
        return X_windowed
