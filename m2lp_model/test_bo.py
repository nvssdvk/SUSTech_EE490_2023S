import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


space = [Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
         Categorical([1e-4, 1e-3, 1e-2, 1e-1], name='learning_rate')]


@use_named_args(space)
def objective(**params):
    model = MLP(X_train.shape[1], 100, len(set(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(torch.Tensor(X_test))
        _, predicted = torch.max(outputs.data, 1)
        equ_arr = np.sum(count_same_elements(predicted, y_test))
        accuracy = equ_arr.item() / len(y_test)
    return -accuracy


def count_same_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2)


if __name__ == "__main__":
    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
# %%
    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
    - activation=%s
    - learning_rate=%.6f""" % (res_gp.x[0], res_gp.x[1]))
