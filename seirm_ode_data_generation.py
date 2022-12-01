import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def seirm_ode(model_params, initial_conditions, max_time):
    s0, e0, i0, r0, m0 = initial_conditions
    N = s0 + e0 + i0 + r0 + m0

    if type(model_params["beta"]) != list:
        model_params["beta"] = [model_params["beta"] for t in range(max_time)]
    if type(model_params["alpha"]) != list:
        model_params["alpha"] = [model_params["alpha"] for t in range(max_time)]
    if type(model_params["gamma"]) != list:
        model_params["gamma"] = [model_params["gamma"] for t in range(max_time)]
    if type(model_params["mu"]) != list:
        model_params["mu"] = [model_params["mu"] for t in range(max_time)]

    ts = {}
    ts["S"], ts["E"], ts["I"], ts["R"], ts["M"] = [s0], [e0], [i0], [r0], [m0]

    for t in range(max_time):
        dS = -model_params["beta"][t] * ts["S"][-1] * ts["I"][-1] / N
        dE = model_params["beta"][t] * ts["S"][-1] * ts["I"][-1] / N - model_params["alpha"][t] * ts["E"][-1]
        dI = model_params["alpha"][t] * ts["E"][-1] - (model_params["gamma"][t] + model_params["mu"][t]) * ts["I"][-1]
        dR = model_params["gamma"][t] * ts["I"][-1]
        dM = model_params["mu"][t] * ts["I"][-1]

        ts["S"].append(ts["S"][-1] + dS)
        ts["E"].append(ts["E"][-1] + dE)
        ts["I"].append(ts["I"][-1] + dI)
        ts["R"].append(ts["R"][-1] + dR)
        ts["M"].append(ts["M"][-1] + dM)

    return ts, N

def linear_parameter_time_series(init_val, final_val, max_time):
    return list(np.arange(init_val, final_val, (final_val - init_val) / max_time))

def sigmoid_parameter_time_series(init_val, final_val, max_time, const):
    return list(init_val + (final_val - init_val) * 1 / (1 + np.exp(-np.arange(-const, const, 2 * const / max_time))))

def get_epiweek_data(max_time):
    epiweeks = []
    year = 2020
    epiweek_num = 10

    for i in range(max_time):
        if epiweek_num >= 52:
            epiweek_num = 0
            year += 1
        epiweeks.append(str(year) + str(int(epiweek_num + i // 7)))
    return epiweeks

def model_params_to_json(model_params, max_time):
    data = {}
    for i in range(max_time):
        data["ode_" + str(i)] = {}
        for param in ["alpha", "beta", "gamma", "mu"]:
            if type(model_params[param]) == float:
                data["ode_" + str(i)][param] = model_params[param]
            else:
                data["ode_" + str(i)][param] = model_params[param][i]
    return json.dumps(data)

def plot_time_series(ts, N, title=None):
    for label, time_series in ts.items():
        plt.plot(np.array(time_series) / N, label=label + "(t)")

    plt.ylabel("Proportion of Population")
    plt.xlabel("Time Step")
    plt.legend()
    
    if title is not None:
        plt.title(title)
    plt.show()

def plot_time_series_with_params(ts, N, model_params, title=None):
    fig, ax = plt.subplots(2)

    for label, time_series in ts.items():
        ax[0].plot(np.array(time_series) / N, label=label + "(t)")

    ax[0].set_ylabel("Proportion of Population")
    ax[0].legend()

    for label, time_series in model_params.items():
        if label == "beta":
            letter = "β"
        if label == "alpha":
            letter = "α"
        if label == "gamma":
            letter = "γ"
        if label == "mu":
            letter = "µ"
        ax[1].plot(time_series, label=letter + "(t)")

    ax[1].set_ylabel("Parameter Value")
    ax[1].set_ylim(0, 0.45)
    ax[1].set_xlabel("Time Step")
    ax[1].legend()

    if title is not None:
        ax[0].set_title(title)

    plt.show()

if __name__ == "__main__":
    max_time = 200

    model_params = {
        "beta": 0.3,
        "alpha": linear_parameter_time_series(0.1, 0.3, max_time),
        "gamma": 0.03,
        "mu": 0.01,
    }
    file_name = "dataset.csv"

    initial_conditions = (900, 0, 100, 0, 0)
    ts, N = seirm_ode(model_params, initial_conditions, max_time)

    # plot_time_series(ts, N, title="SEIRM (β=0.3, α=0.2, γ=0.05, µ=0.02)")
    # plot_time_series_with_params(ts, N, model_params, title="SEIRM (β varies, α=0.2, γ varies, µ=0.02)")

    data = pd.DataFrame(ts)
    data.to_csv(file_name, index=False)