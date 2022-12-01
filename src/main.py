import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

data_directory = "data"
aggregated_losses = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

for file in os.listdir(data_directory):
    file = os.path.join(data_directory, file)
    if file.endswith(".csv"):
        data = pd.read_csv(file)

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

            aggregated_losses[i].append(losses[-1])

for index, losses in aggregated_losses.items():
    print("Mean Loss for " + str(index) + ": " + str(np.mean(losses)))
