"""
Mortality forecasting for the Swiss population (Human Mortality Database,
ages 0-99, calendar years 1950-2016): Lee-Carter vs. recurrent neural networks.

Python re-implementation and extension of course Assignment 1 (original R
version in original-R/):
  GRU with three hidden layers (20 / 15 / 10 units), Adam, batch size 100,
  500 epochs, 20% of the training samples held back to monitor over-fitting,
  trained on 1950-1999 and used to project 2000-2016 recursively (male).
Extensions (not in the original assignment):
  - Lee-Carter (SVD + random walk with drift) benchmark for both genders
  - LSTM with the same architecture, and female as well as male mortality
  - several random seeds per network and an ensemble (average) forecast,
    because single runs are sensitive to the initialisation

Loss = 10^4 x mean squared error of the mortality rates m_x (the course
metric); RMSE of log-mortality is reported as well.

Usage:  python mortality_forecast.py data/mortality_CHE.csv results [n_seeds]
"""
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import keras
from keras import layers

T0, TAU0 = 10, 5            # look-back years, age window
OBS_YEAR, LAST_YEAR = 1999, 2016
UNITS = (20, 15, 10)
EPOCHS, BATCH = 500, 100


def rate_matrix(df, gender):
    m = df[df.Gender == gender].pivot(index="Year", columns="Age", values="logmx").sort_index()
    return m.index.to_numpy(), m.to_numpy(float)          # years, (years x ages) log m_x


def pad_ages(a):
    d = (TAU0 - 1) // 2
    return np.concatenate([np.repeat(a[:, :1], d, 1), a, np.repeat(a[:, -1:], d, 1)], 1)


def make_samples(logm):
    """x = (T0 years x TAU0 neighbouring ages) window, y = next year's rate at the centre age."""
    p = pad_ages(logm)
    t1, a1 = logm.shape[0] - T0, logm.shape[1]
    x = np.empty((t1 * a1, T0, TAU0), "float32")
    y = np.empty(t1 * a1, "float32")
    for t in range(t1):
        for a in range(a1):
            x[t * a1 + a] = p[t:t + T0, a:a + TAU0]
            y[t * a1 + a] = logm[t + T0, a]
    return x, y


def build(kind, y0):
    cell = layers.GRU if kind == "GRU" else layers.LSTM
    inp = keras.Input((T0, TAU0))
    h = inp
    for i, u in enumerate(UNITS):
        h = cell(u, activation="tanh", recurrent_activation="tanh",
                 return_sequences=i < len(UNITS) - 1)(h)
    out = layers.Dense(1, activation=keras.ops.exp, kernel_initializer="zeros",
                       bias_initializer=keras.initializers.Constant(float(np.log(y0))))(h)
    model = keras.Model(inp, out)
    model.compile(loss="mse", optimizer="adam", jit_compile=True, steps_per_execution=32)  # speed only
    return model


def scale(x, lo, hi):          # same min-max transform as the course code
    return 2 * (x - lo) / (lo - hi) - 1


def recursive_forecast(model, years, logm, lo, hi):
    work = logm.copy()
    for yr in range(OBS_YEAR + 1, LAST_YEAR + 1):
        i = int(np.where(years == yr)[0][0])
        x = pad_ages(work[i - T0:i])
        x = np.stack([x[:, a:a + TAU0] for a in range(logm.shape[1])]).astype("float32")
        work[i] = -model.predict(scale(x, lo, hi), verbose=0).ravel()
    return work[years > OBS_YEAR]


def lee_carter(years, logm):
    tr = logm[years <= OBS_YEAR]
    ax = tr.mean(0)
    u, s, vt = np.linalg.svd((tr - ax).T, full_matrices=False)
    bx, kt = u[:, 0] * s[0], vt[0]
    c1, c2 = kt.mean(), bx.sum()
    ax, bx, kt = ax + c1 * bx, bx / c2, (kt - c1) * c2
    drift = (kt[-1] - kt[0]) / (len(kt) - 1)
    h = np.arange(1, (years > OBS_YEAR).sum() + 1)
    return ax + np.outer(kt, bx), ax + np.outer(kt[-1] + drift * h, bx)


def metrics(pred_log, obs_log):
    return dict(mse_mx_1e4=float(1e4 * np.mean((np.exp(pred_log) - np.exp(obs_log)) ** 2)),
                rmse_log=float(np.sqrt(np.mean((pred_log - obs_log) ** 2))))


def main(csv, out, n_seeds):
    t_start = time.time()
    df = pd.read_csv(csv, sep=";")
    res = dict(settings=dict(train="1950-1999", validation="2000-2016", ages="0-99", lookback=T0,
                             age_window=TAU0, units=UNITS, epochs=EPOCHS, batch=BATCH, seeds=n_seeds))
    fc_rows = []
    for gender in ["Male", "Female"]:
        years, logm = rate_matrix(df, gender)
        obs_val = logm[years > OBS_YEAR]
        vyears = years[years > OBS_YEAR]
        g = {}

        fit_lc, fc_lc = lee_carter(years, logm)
        g["Lee-Carter"] = dict(in_sample=metrics(fit_lc, logm[years <= OBS_YEAR]),
                               out_sample=metrics(fc_lc, obs_val),
                               by_year={int(y): metrics(fc_lc[i], obs_val[i])["mse_mx_1e4"]
                                        for i, y in enumerate(vyears)})
        forecasts = {"Lee-Carter": fc_lc}

        x, y = make_samples(logm[years <= OBS_YEAR])
        lo, hi = x.min(), x.max()
        xs, ys = scale(x, lo, hi), -y
        for kind in ["GRU", "LSTM"]:
            runs, fcs = [], []
            for seed in range(n_seeds):
                keras.utils.set_random_seed(1234 + seed)
                model = build(kind, ys.mean())
                ck = f"{out}/_best.weights.h5"
                hist = model.fit(xs, ys, validation_split=0.2, batch_size=BATCH, epochs=EPOCHS, verbose=0,
                                 callbacks=[keras.callbacks.ModelCheckpoint(
                                     ck, monitor="val_loss", save_best_only=True, save_weights_only=True)])
                vl = np.array(hist.history["val_loss"])
                model.load_weights(ck)
                fit = -model.predict(xs, verbose=0, batch_size=1000).ravel()
                fc = recursive_forecast(model, years, logm, lo, hi)
                fcs.append(fc)
                runs.append(dict(seed=1234 + seed, best_epoch=int(vl.argmin()) + 1,
                                 keras_train_loss_at_best=float(hist.history["loss"][int(vl.argmin())]),
                                 keras_val_loss_best=float(vl.min()),
                                 keras_train_loss_final=float(hist.history["loss"][-1]),
                                 keras_val_loss_final=float(vl[-1]),
                                 in_sample=metrics(fit, y), out_sample=metrics(fc, obs_val),
                                 loss_2012=metrics(fc[vyears == 2012], obs_val[vyears == 2012])["mse_mx_1e4"],
                                 loss_2013=metrics(fc[vyears == 2013], obs_val[vyears == 2013])["mse_mx_1e4"]))
                if seed == 0:
                    pd.DataFrame(dict(epoch=np.arange(1, EPOCHS + 1), loss=hist.history["loss"],
                                      val_loss=vl)).to_csv(f"{out}/loss_curve_{kind}_{gender}.csv", index=False)
                print(gender, kind, runs[-1]["seed"], runs[-1]["best_epoch"], runs[-1]["out_sample"],
                      round(time.time() - t_start), flush=True)
            ens = np.mean(fcs, 0)                       # average of log-rates across seeds
            outs = [r["out_sample"]["mse_mx_1e4"] for r in runs]
            g[kind] = dict(runs=runs,
                           out_sample_mean=float(np.mean(outs)), out_sample_sd=float(np.std(outs, ddof=1)) if n_seeds > 1 else None,
                           out_sample_min=float(np.min(outs)), out_sample_max=float(np.max(outs)),
                           in_sample_mean=float(np.mean([r["in_sample"]["mse_mx_1e4"] for r in runs])),
                           ensemble_out_sample=metrics(ens, obs_val),
                           ensemble_by_year={int(yv): metrics(ens[i], obs_val[i])["mse_mx_1e4"]
                                             for i, yv in enumerate(vyears)})
            forecasts[f"{kind} ensemble"] = ens
            forecasts[f"{kind} seed 1234"] = fcs[0]
        res[gender] = g
        for name, fc in forecasts.items():
            for i, yv in enumerate(vyears):
                for a in range(fc.shape[1]):
                    fc_rows.append((gender, name, int(yv), a, float(obs_val[i, a]), float(fc[i, a])))
        json.dump(res, open(f"{out}/mortality_results.json", "w"), indent=2)
    pd.DataFrame(fc_rows, columns=["Gender", "Model", "Year", "Age", "logmx_observed", "logmx_forecast"]
                 ).to_csv(f"{out}/mortality_forecasts_2000_2016.csv", index=False)
    if os.path.exists(f"{out}/_best.weights.h5"):
        os.remove(f"{out}/_best.weights.h5")
    res["runtime_sec"] = round(time.time() - t_start)
    json.dump(res, open(f"{out}/mortality_results.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
