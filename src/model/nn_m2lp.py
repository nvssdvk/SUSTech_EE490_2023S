# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import pandas as pd
import numpy as np
import os

# For plotting
import matplotlib
import matplotlib.pyplot as plt

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


def plot_loss(loss_record):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['verify'])]
    plt.figure()
    plt.plot(x_1, (loss_record['train']), c='tab:red', label='train')
    plt.plot(x_2, (loss_record['verify']), c='tab:cyan', label='verify')
    # plt.plot(x_1, np.rad2deg(loss_record['train']), c='tab:red', label='train')
    # plt.plot(x_2, np.rad2deg(loss_record['verify']), c='tab:cyan', label='verify')
    plt.ylim(0.0, 120)
    plt.xlabel('Training steps')
    plt.ylabel('MAE')
    plt.title('MAE Loss Curve')
    plt.legend()
    plt.savefig(r"../../img/nn_model/loss.png")
    plt.show()


def plot_pred(ve_set, model, device, preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in ve_set:
            x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    preds = preds.reshape(len(preds), 1)
    targets = targets.reshape(len(targets), 1)

    # preds = np.rad2deg(preds)
    # targets = np.rad2deg(targets)

    plt.figure()
    plt.scatter(targets, preds, c='r', alpha=0.5, label="Predicted data")
    plt.plot([-900, 180], [-900, 180], c='b', label="Theoretical data")
    plt.xlim(-900, 180)
    plt.ylim(-900, 180)
    plt.xlabel('Phase by CST')
    plt.ylabel('Phase by M2LP')
    plt.title('Prediction Error Curve')
    plt.legend(loc='best')
    plt.savefig(r"../../img/nn_model/prediction_error.png")
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
            # data_min = np.min(data, axis=0)
            # data_max = np.max(data, axis=0)
            # self.data = (self.data - data_min) / (data_max - data_min)

        else:
            target = (data[:, -1])
            # target = np.deg2rad(data[:, -1])
            data = data[:, feats]
            # indices = torch.zeros([len(data), 1])
            # if mode == 'train':
            #     indices = [i for i in range(len(data)) if i % 10 != 0]
            # elif mode == 'verify':
            #     indices = [i for i in range(len(data)) if i % 10 == 0]
            # self.data = data[indices]
            # self.target = target[indices]
            self.data = data
            self.target = target

            # data_min = np.min(data, axis=0)
            # data_max = np.max(data, axis=0)
            # target_min = np.min(target, axis=0)
            # target_max = np.max(target, axis=0)
            # self.data = (self.data - data_min) / (data_max - data_min)
            # self.target = (self.target - target_min) / (target_max - target_min)

            self.data = torch.FloatTensor(self.data)
            self.target = torch.FloatTensor(self.target)

        self.dim = self.data.shape[1]
        print(f'Finished reading the {mode} set, ({len(self.data)} samples found, dim = {self.dim})')
        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ['train', 'verify']:
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

    def __init__(self, input_dim, first_dim, block_num, alpha):
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


def train(tr_set, ve_set, model, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    n_epochs = config['n_epochs']
    min_loss = config['min_loss']
    loss_record = {'train': [], 'verify': []}
    early_stop_cnt = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
            pred = model(x)
            batch_loss = model.cal_loss(pred, y)
            batch_loss.backward()
            optimizer.step()
            loss_record['train'].append(batch_loss.detach().cpu().item())
            epoch_loss += batch_loss.detach().cpu().item() * len(x)
        scheduler.step()
        epoch_loss /= len(tr_set.dataset)
        print(f'Training[{epoch + 1}/{n_epochs}], Loss={epoch_loss}', end='\r')

        model.eval()
        ver_los = 0
        for x, y in ve_set:
            x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
            with torch.no_grad():
                pred = model(x)
                batch_loss = model.cal_loss(pred, y)
                loss_record['verify'].append(batch_loss.detach().cpu().item())
            ver_los += batch_loss.detach().cpu().item() * len(x)
        ver_los /= len(ve_set.dataset)

        if ver_los < min_loss:
            min_loss = ver_los
            print(f'Saving model (epoch = {epoch + 1}, loss = {min_loss})')
            torch.save(model.state_dict(), config['model_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # loss_record['verify'].append(ver_los)
        if early_stop_cnt > config['early_stop']:
            print(f'Finished training after {epoch + 1} epochs')
            break
        elif epoch == n_epochs - 1:
            print(f'Finished training after {epoch + 1} epochs')

    return min_loss, loss_record


def verify(ve_set, model, device):
    model.eval()
    ver_los = 0
    for x, y in ve_set:
        x, y = x.reshape(len(x), 3, 1).to(device), y.reshape(len(y), 1, 1).to(device)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)
            batch_loss = model.cal_loss(pred, y)
        ver_los += batch_loss.detach().cpu().item() * len(x)  # accumulate loss
        # loss_record['verify'].append(ver_los)
    ver_los /= len(ve_set.dataset)

    return ver_los


def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.reshape(len(x), 3, 1).to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)
            preds.append(pred.detach().cpu())  # collect prediction
    preds = torch.cat(preds, dim=0).numpy()  # concatenate all predictions and convert to a numpy array
    return preds


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    pred_name = ["a", "h", "e", "pred angle"]
    pred_set = np.zeros([len(preds), 4])
    temp_csv = pd.read_csv(r"../../data/dataset/space.csv", header=0, engine="c").values
    pred_set[:, 0:3] = temp_csv[:, 1:]
    pred_set[:, -1] = preds[:, 0, 0]
    df = pd.DataFrame(columns=pred_name, data=pred_set)
    df.to_csv(r'../../data/dataset/pred_set.csv', encoding='utf-8', index=False)

    data = pd.read_csv(r'../../data/dataset/pred_set.csv', header=0, engine="c").values

    fig_unwrap_scatter = plt.figure(num=1, figsize=(19.2, 10.8))
    fig_unwrap_tricontourf = plt.figure(num=2, figsize=(19.2, 10.8))
    para_e = np.unique(data[:, 2])
    for e in para_e:
        data_copy_at_e = data[(data[:, 2] == e)]
        sort_index = np.lexsort((data_copy_at_e[:, 0], data_copy_at_e[:, 1]))
        data_copy_at_e = data_copy_at_e[sort_index]

        plt.figure(1)
        plt.subplot(2, 2, np.where(para_e == e)[0].item() + 1)
        plt.scatter(data_copy_at_e[:, 0], data_copy_at_e[:, 1], c=data_copy_at_e[:, 3], cmap='jet')
        plt.colorbar()
        plt.xlabel("a (Top Surface Length)")
        plt.ylabel("h (Height)")
        plt.title("Wraped Phase at e={:.2f}".format(e))

        plt.figure(2)
        plt.subplot(2, 2, np.where(para_e == e)[0].item() + 1)
        plt.tricontourf(data_copy_at_e[:, 0], data_copy_at_e[:, 1], data_copy_at_e[:, 3], cmap='jet')
        plt.colorbar()
        plt.xlabel("a (Top Surface Length)")
        plt.ylabel("h (Height)")
        plt.title("Wraped Phase at e={:.2f}".format(e))
    plt.figure(1)
    plt.savefig(r"../../img/tt_set_phase/fig_wrap_scatter.png")
    plt.figure(2)
    plt.savefig(r"../../img/tt_set_phase/fig_wrap_tricontourf.png")
    plt.show()


if __name__ == "__main__":
    config = {
        # dataset
        'batch_size': 500,
        # model
        'block_num': 3,
        'first_dim': 64,  # 16,128,
        'alpha': 0.04,
        # Adam
        'learning_rate': 1e-2,
        # train
        'n_epochs': 1000,
        'early_stop': 200,
        'min_loss': 1000.,
        # path
        'model_path': 'models/model.pth',
        'tr_path': r'../../data/dataset/tr_set_unwrap.csv',
        've_path': r'../../data/dataset/ve_set_unwrap.csv',
        'tt_path': r'../../data/dataset/tt_set.csv'
    }

    device = get_device()

    tr_set = prep_dataloader(config['tr_path'], 'train', config['batch_size'])
    ve_set = prep_dataloader(config['ve_path'], 'verify', config['batch_size'])
    tt_set = prep_dataloader(config['tt_path'], 'test', config['batch_size'])

    # model = M2LP(alpha=config['alpha'], input_dim=tr_set.dataset.dim, block_num=config['block_num'],
    #              first_dim=config['first_dim']).to(device)
    # model_loss, model_loss_record = train(tr_set, ve_set, model, config, device)
    # # # %%
    # plot_loss(model_loss_record)
    # del model

    model = M2LP(alpha=config['alpha'], input_dim=tr_set.dataset.dim, block_num=config['block_num'],
                 first_dim=config['first_dim']).to(device)
    ckpt = torch.load(config['model_path'], map_location='cpu')
    model.load_state_dict(ckpt)
    plot_pred(ve_set, model, device)

    # # %%
    preds = test(tt_set, model, device)
    save_pred(preds, '../data/dataset/pred.csv')
