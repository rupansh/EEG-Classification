from glob import glob
import pandas as pd
import numpy as np

def load_training_sk():
    ts = []
    gt = []
    vts = []
    vgt = []
    for fn in glob("./data/train/*data.csv"):
        ev_fn = fn.replace("data.csv", "events.csv")
        x = pd.read_csv(fn, index_col=0).values.astype(np.float32)
        y = pd.read_csv(ev_fn, index_col=0).values.astype(np.float32)
        ts.append(x)
        gt.append(y)
    vts = ts[-2:]
    vgt = gt[-2:]
    ts = ts[:-2]
    gt = gt[:-2]

    return ((ts, gt), (vts, vgt))

def load_training_data():
    ts = []
    gt = []
    validation_ts = []
    validation_gt = []
    for fn in glob("./data/train/*data.csv"):
        ev_fn = fn.replace("data.csv", "events.csv")
        x = pd.read_csv(fn).iloc[:,1:].values
        y = pd.read_csv(ev_fn).iloc[:,1:].values
        ts.append(x.T.astype(np.float32))
        gt.append(y.T.astype(np.float32))

    validation_ts = ts[-2:]
    validation_gt = gt[-2:]
    ts = ts[:-2]
    gt = gt[:-2]

    return ((ts, gt), (validation_ts, validation_gt))
