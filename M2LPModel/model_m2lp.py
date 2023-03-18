# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For Data preprocess
import pandas as pd
import numpy as np
import os

# For plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

matplotlib.rcParams['axes.unicode_minus'] = False

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure()
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 120)
    plt.xlabel('Training steps')
    plt.ylabel('MAE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure()
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-180, 180], [-180, 180], c='b')
    plt.xlim(-220, 220)
    plt.ylim(-220, 220)
    # plt.xlabel('Target')
    # plt.ylabel('Predicted')
    plt.title('Target v.s. Prediction')
    plt.show()


class M2LPDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode

        data = pd.read_csv(data_dir, header=0, engine="c").values
        feats = list(range(3))
        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
            # data_min = np.min(Data, axis=0)
            # data_max = np.max(Data, axis=0)
            # self.Data = (self.Data - data_min) / (data_max - data_min)

        else:
            target = data[:, -1]
            data = data[:, feats]
            indices = torch.zeros([len(data), 1])
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            self.data = data[indices]
            self.target = target[indices]

            # data_min = np.min(Data, axis=0)
            # data_max = np.max(Data, axis=0)
            # target_min = np.min(target, axis=0)
            # target_max = np.max(target, axis=0)
            # self.Data = (self.Data - data_min) / (data_max - data_min)
            # self.target = (self.target - target_min) / (target_max - target_min)

            self.data = torch.FloatTensor(self.data)
            self.target = torch.FloatTensor(self.target)

        self.dim = self.data.shape[1]
        print(
            f'Finished reading the {mode} set of Dataset ({len(self.data)} samples found, dim = {self.dim})')
        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]


def prep_dataloader(data_dir, mode, batch_size: int, n_jobs=0):
    dataset = M2LPDataset(data_dir, mode=mode)  # Construct dataset
    dataloader = DataLoader(dataset, batch_size,
                            shuffle=(mode == 'train'), drop_last=False, num_workers=n_jobs, pin_memory=True)
    return dataloader


class M2LP(nn.Module):

    def __init__(self, input_dim=1, first_dim=1, block_num=1, alpha=0.01):
        super(M2LP, self).__init__()

        self.prep_layer = nn.Sequential(
            # nn.Linear(input_dim, first_dim),
            nn.Conv1d(input_dim, first_dim, kernel_size=1),
            nn.BatchNorm1d(first_dim),
            nn.LeakyReLU(alpha)
        )

        self.regr_layer = nn.Sequential(
            # nn.Linear(first_dim * pow(2, block_num), 1)
            nn.Conv1d(first_dim * pow(2, block_num), 1, 1),
        )

        self.hyper_layers = nn.Sequential()
        temp_dim = first_dim
        for block_id in range(block_num):
            # self.hyper_layers.add_module(name=f'L{block_id}', module=nn.Linear(temp_dim, temp_dim))
            self.hyper_layers.add_module(name=f'Conv{block_id}', module=nn.Conv1d(temp_dim, temp_dim, 1))
            self.hyper_layers.add_module(name=f'BN{block_id}', module=nn.BatchNorm1d(temp_dim))
            self.hyper_layers.add_module(name=f'ReLu{block_id}', module=nn.LeakyReLU(alpha))
            if block_id != (block_num - 1):
                # self.hyper_layers.add_module(name=f'L{block_id}_exp', module=nn.Linear(temp_dim, temp_dim * 2))
                self.hyper_layers.add_module(name=f'Conv{block_id}_exp', module=nn.Conv1d(temp_dim, temp_dim * 2, 1))
                self.hyper_layers.add_module(name=f'BN{block_id}_exp', module=nn.BatchNorm1d(temp_dim * 2))
                self.hyper_layers.add_module(name=f'ReLu{block_id}_exp', module=nn.LeakyReLU(alpha))
                temp_dim = temp_dim * 2
        # self.hyper_layers.add_module(name=f'L_post', module=nn.Linear(temp_dim, temp_dim * 2))
        self.hyper_layers.add_module(name=f'Conv_post', module=nn.Conv1d(temp_dim, temp_dim * 2, 1))
        self.hyper_layers.add_module(name=f'BN_post', module=nn.BatchNorm1d(temp_dim * 2))
        self.hyper_layers.add_module(name=f'ReLu_post', module=nn.LeakyReLU(alpha))
        temp_dim = first_dim

        self.criterion = nn.L1Loss(reduction='mean')
        # self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.hyper_layers(out)
        out = self.regr_layer(out)
        return out

    def cal_loss(self, pred, target):
        """ Calculate loss """
        return self.criterion(pred, target)


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # optimizer = torch.optim.SGD(M2LPModel.parameters(), lr=config['learning_rate'], momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    min_loss = config['min_loss']

    loss_record = {'train': [], 'dev': []}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        epoch_loss = 0
        model.train()  # set M2LPModel to training mode
        for x, y in tr_set:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
            pred = model(x)  # forward pass (compute output)
            batch_loss = model.cal_loss(pred, y)  # compute loss
            batch_loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update M2LPModel with optimizer
            loss_record['train'].append(batch_loss.detach().cpu().item())
            epoch_loss += batch_loss.detach().cpu().item()
        scheduler.step()
        print(f'Training[{epoch + 1}/{n_epochs}], Loss={epoch_loss / len(tr_set.dataset)}', end='\r')

        # After each epoch, test your M2LPModel on the validation (development) set.
        dev_loss = dev(dv_set, model, device)
        if dev_loss < min_loss:
            # Save M2LPModel if your M2LPModel improved
            min_loss = dev_loss
            print(f'Saving model (epoch = {epoch + 1}, loss = {min_loss})')
            torch.save(model.state_dict(), config['model_path'])  # Save M2LPModel to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_loss)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your M2LPModel stops improving for "config['early_stop']" epochs.
            break

    print(F'Finished training after {epoch} epochs')
    return min_loss, loss_record


def dev(dv_set, model, device):
    model.eval()  # set M2LPModel to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            batch_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += batch_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss /= len(dv_set.dataset)  # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()  # set M2LPModel to evalutation mode
    preds = []
    for x in tt_set:  # iterate through the dataloader
        x = x.reshape(len(x), 3, 1).to(device)  # move Data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    pred_name = ["a", "h", "e", "pred angle"]
    pred_set = np.zeros([len(preds), 4])
    temp_csv = pd.read_csv(r"../Data/dataset/space.csv", header=0, engine="c").values
    pred_set[:, 0:3] = temp_csv[:, 1:]
    pred_set[:, -1] = preds[:, 0, 0]
    df = pd.DataFrame(columns=pred_name, data=pred_set)
    df.to_csv(f'../Data/dataset/pred_set.csv', encoding='utf-8', index=False)
    del df


if __name__ == "__main__":
    config = {
        # dataset
        'batch_size': 400,
        # M2LPModel
        'block_num': 4,
        'first_dim': 32,
        'alpha': 0.03,
        # Adam
        'weight_decay': 0,
        'learning_rate': 1e-2,
        # train
        'n_epochs': 1000,
        'early_stop': 300,
        'min_loss': 1000.,
        # path
        'model_path': 'models/m2lp_model.pth',
        'tr_path': '../Data/dataset/tr_set.csv',
        'tt_path': '../Data/dataset/tt_set.csv'
    }

    device = get_device()  # get the current available device ('cpu' or 'cuda')

    tr_set = prep_dataloader(config['tr_path'], 'train', config['batch_size'])
    dv_set = prep_dataloader(config['tr_path'], 'dev', config['batch_size'])
    tt_set = prep_dataloader(config['tt_path'], 'test', config['batch_size'])
    # M2LPModel = M2LP(alpha=config['alpha'], input_dim=tr_set.dataset.dim, block_num=config['block_num'],
    #              first_dim=config['first_dim']).to(device)  # Construct M2LPModel and move to device
    # model_loss, model_loss_record = train(tr_set, dv_set, M2LPModel, config, device)
    # # %%
    # plot_learning_curve(model_loss_record, title='m2lp M2LPModel')
    # del M2LPModel
    model = M2LP(alpha=config['alpha'], input_dim=tr_set.dataset.dim, block_num=config['block_num'],
                 first_dim=config['first_dim']).to(device)  # Construct M2LPModel and move to device
    ckpt = torch.load(config['model_path'], map_location='cpu')  # Load your best M2LPModel
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)
    # %%
    preds = test(tt_set, model, device)  # predict COVID-19 cases with your M2LPModel
    save_pred(preds, '../Data/dataset/pred.csv')