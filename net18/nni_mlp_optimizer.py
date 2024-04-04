import torch
import torch.nn as nn
import nni
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

class ANN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # self.bn1 = nn.BatchNorm1d(n_input)
        self.hidden = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        #self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = torch.sigmoid(self.hidden(x))

        #x = self.bn1(x)

        x = self.hidden(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.output(x)
        return x


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

device = ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device}")



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #print(f'size: {size}')

    model.train()
    l = None
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #print(X.detach().numpy()[0])
        #raise Exception("FORCED EXC")
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch == (len(dataloader) - 1):
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            l = loss
    return l

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss


'''
def test(dataloader, model, loss_fn):
    def get_data_by_scenario_and_case(scenario, case):
        assert case > 0
        assert scenario > 0

        file_excel = f'./validation_data/Scenario{scenario}.xlsx'
        xls = pd.ExcelFile(file_excel)
        scheda = xls.sheet_names[case - 1]
        dati_scheda = pd.read_excel(file_excel, sheet_name=scheda)
        valori_veri_misure = dati_scheda['Valori veri misure'].tolist()
        misure = dati_scheda['Misure'].tolist()
        valori_veri_stima = dati_scheda['Valori veri stima'].tolist()
        valori_stimati = dati_scheda['Valori stimati'].tolist()
        x = np.hstack((
            [list(-1. * np.asarray(valori_veri_misure[5:22]))], [list(-1. * np.asarray(valori_veri_misure[22:39]))],
            [valori_veri_misure[:5]], [valori_veri_misure[39:46]], [valori_veri_misure[46:]]
        ))

        x_hat = np.hstack((
            [list(-1. * np.asarray(misure[5:22]))], [list(-1. * np.asarray(misure[22:39]))], [misure[:5]],
            [misure[39:46]], [misure[46:]]
        ))
        y = [valori_veri_stima[:18]]
        y_hat = [valori_stimati[:18]]

        sens_tab = pd.read_excel(file_excel, sheet_name='Sensitivity')
        sens = []

        # sens = np.hstack((s1_sens_pi, s1_sens_qi, s1_sens_vmpu, s1_sens_pf, s1_sens_qf))
        return x, x_hat, y, y_hat, sens

    data_x = np.load('../nets/net_18_data/measured_data_x.npy')
    data_y = np.load('../nets/net_18_data/data_y.npy')

    data_x = np.delete(data_x, 18, axis=1)
    data_x = np.delete(data_x, 0, axis=1)

    model.eval()
    model = model.to(device)

    OUTPUT = .0
    print('SCENARIO 1, CASE 1 VALIDATION')
    s1_c1_data = get_data_by_scenario_and_case(1, 1)
    x = s1_c1_data[0]
    x_hat = s1_c1_data[1]
    y = s1_c1_data[2]
    y_hat = s1_c1_data[3]


    test_data = Dataset(x, y)
    # test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        to_be_explained = X
        pred = model(X)
        y = y.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        print(f'y: {y}\npred: {pred}')
        print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
        OUTPUT += np.sqrt(np.mean(np.square(y - pred)))

    # print('SCENARIO 2, CASE 1 VALIDATION')
    s1_c1_data = get_data_by_scenario_and_case(2, 1)
    x = s1_c1_data[0]
    x_hat = s1_c1_data[1]
    y = s1_c1_data[2]
    y_hat = s1_c1_data[3]

    test_data = Dataset(x, y)
    # test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            # print(f'y: {y}\npred: {pred}')
            print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
            OUTPUT += np.sqrt(np.mean(np.square(y - pred)))

    # print('SCENARIO 3, CASE 1 VALIDATION')
    s1_c1_data = get_data_by_scenario_and_case(3, 1)
    x = s1_c1_data[0]
    x_hat = s1_c1_data[1]
    y = s1_c1_data[2]
    y_hat = s1_c1_data[3]

    test_data = Dataset(x, y)
    # test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            # print(f'y: {y}\npred: {pred}')
            print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
            OUTPUT += np.sqrt(np.mean(np.square(y - pred)))

    # print('SCENARIO 4, CASE 1 VALIDATION')
    s1_c1_data = get_data_by_scenario_and_case(4, 1)
    x = s1_c1_data[0]
    x_hat = s1_c1_data[1]
    y = s1_c1_data[2]
    y_hat = s1_c1_data[3]

    test_data = Dataset(x, y)
    # test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            # print(f'y: {y}\npred: {pred}')
            print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
            OUTPUT += np.sqrt(np.mean(np.square(y - pred)))

    # print('SCENARIO 5, CASE 1 VALIDATION')
    s1_c1_data = get_data_by_scenario_and_case(5, 1)
    x = s1_c1_data[0]
    x_hat = s1_c1_data[1]
    y = s1_c1_data[2]
    y_hat = s1_c1_data[3]

    test_data = Dataset(x, y)
    # test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            # print(f'y: {y}\npred: {pred}')
            print(f'std: {np.sqrt(np.mean(np.square(y - pred)))}')
            OUTPUT += np.sqrt(np.mean(np.square(y - pred)))

    return OUTPUT
'''

def main_mlp_optimization(params):
    alt_x = np.load('../nets/net_18_data/measured_data_x_alt.npy')
    alt_y = np.load('../nets/net_18_data/data_y_alt.npy')

    data_x = alt_x
    data_y = alt_y


    # separate them into training 60%, test 40%
    split_train = int(0.8 * data_x.shape[0])
    train_x = data_x[:split_train, :]
    train_y = data_y[:split_train, :]

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True, random_state=42)

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3, shuffle=True, random_state=42)


    batch_size = params['batch_size']

    train_data = Dataset(train_x, train_y)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)

    val_data = Dataset(val_x, val_y)

    test_data = Dataset(test_x, test_y)
    # test_dataloader = DataLoader(test_data, batch_size=100, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    # Train the model
    input_shape = train_x.shape[1]
    num_classes = train_y.shape[1]

    units = params['units']
    dropout = params['dropout']
    nni_epochs = params['epochs']
    lr = params['lr']

    model = ANN(input_shape, units, num_classes, dropout=dropout).to(device)
    model.to(device)
    epochs = nni_epochs
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # mse_sum = 100
    mse = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        mse = test(test_dataloader, model, loss_fn)
        #nni.report_intermediate_result(mse_sum)
        nni.report_intermediate_result(mse)
    nni.report_final_result(mse)
    #nni.report_final_result(mse_sum)


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        torch.manual_seed(42)
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main_mlp_optimization(params)
    except Exception as exception:
        # logger.exception(exception)
        raise