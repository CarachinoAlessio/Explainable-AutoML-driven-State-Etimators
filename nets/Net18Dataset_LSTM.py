from torch.utils.data import Dataset


class Net18Dataset_LSTM(Dataset):
    def __init__(self, data_x, data_y, sequence_length):
        #sequence_length = 1
        self.data_x = data_x
        self.data_y = data_y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data_x) - self.sequence_length

    def __getitem__(self, index):
        x = self.data_x[index:index + self.sequence_length]
        y = self.data_y[index + self.sequence_length]
        return x, y
