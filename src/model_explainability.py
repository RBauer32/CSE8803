import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

file_path = "data/dataset0.csv"
data = pd.read_csv(file_path)

data = np.array(data[["S", "E", "I", "R", "M"]])
data = data / np.amax(data)

for i in range(5):
    X, y = np.hstack((data[:, :i], data[:, i + 1:])), data[:, i]

    window_size = 20
    windowed_X = []

    for j in range(len(y) - window_size):
        windowed_X.append(X[j : j + window_size].flatten())

    X = torch.Tensor(windowed_X)
    y = torch.Tensor(y[window_size:])

    model = nn.Sequential(nn.Linear(80, 70),
                          nn.ReLU(),
                          nn.Linear(70, 20),
                          nn.ReLU(),
                          nn.Linear(20, 1),
                          nn.Sigmoid())
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(3000):
        pred_y = model(X)
        loss = loss_function(pred_y, y)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()

    model.eval()
    ig = IntegratedGradients(model)
    input = torch.rand(4000, 80)
    baseline = torch.zeros(4000, 80)

    attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
    print('IG Attributions:', attributions)
    print('Convergence Delta:', delta)

    attributions = attributions.numpy()
    attributions = np.sum(attributions, axis=0)
    attributions /= np.amax(attributions)

    plt.imshow(attributions.reshape(20, 4))
    plt.show()
