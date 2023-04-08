import torch

device = "cpu"
class Dataset(torch.utils.data.Dataset):
    '''
  Prepare the Boston dataset for regression
  '''
    def __init__(self, input, output):
        self.X = torch.tensor(input, dtype=torch.float32).to(device)
        self.Y = torch.tensor(output, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
