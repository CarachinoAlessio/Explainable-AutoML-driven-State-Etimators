import torch

device = "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input, output, x_mean=None, x_std=None, y_mean=None, y_std=None, znorm=False):
        '''
        if x_mean is None:
            x_mean = input.mean(axis=1)
        self.x_mean = x_mean

        if y_mean is None:
            x_mean = input.mean(axis=1)
        self.x_mean = x_mean

        if std is None:
            std = input.std(axis=1)
        self.std = std
        '''
        if not znorm:
            self.X = torch.tensor(input, dtype=torch.float32).to(device)
            self.Y = torch.tensor(output, dtype=torch.float32).to(device)
        else:
            # todo: fix the following code to enable znorm
            self.X = torch.tensor(
                (input-self.mean)/self.std,
                dtype=torch.float32
            ).to(device)
            self.Y = torch.tensor(
                (output - self.imean) / self.std,
                dtype=torch.float32
            ).to(device)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
