"""
Run the original models and get data for training.
"""
import sys
import operator
import os
import numpy as np
import pandas as pd
import time
import json
from operator import itemgetter
import csv
import scipy.stats as stats
from itertools import groupby
from operator import itemgetter
from datetime import datetime as dt

from telemanom._globals import Config
import telemanom.errors as err
import telemanom.helpers as helpers
import telemanom.modeling as models


def run(config, _id, logger):
    ''' Top-level function for running experiment.

    Args:
        config (dict): Parameters for modeling, execution levels, and error calculations loaded from config.yaml
        _id (str): Unique id for each processing run generated from current time
        logger (obj): Logger obj from logging module

    Returns:
        None

    '''

    stats = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0
    }

    with open("labeled_anomalies.csv", "rU") as f:
        reader = csv.DictReader(f)

        with open("results/%s.csv" %_id, "a") as out:

            writer = csv.DictWriter(out, config.header) # line by line results written to csv
            writer.writeheader()
        
            for i, anom in enumerate(reader):
                if reader.line_num >= 1:

                    anom['run_id'] = _id
                    logger.info("Stream # %s: %s" %(reader.line_num-1, anom['chan_id']))
                    model = None

                    X_train, y_train, X_test, y_test = helpers.load_data(anom)
                    
                    # Generate or load predictions
                    # ===============================
                    y_hat = [float(x) for x in list(np.load(os.path.join("data", config.use_id, "y_hat", anom["chan_id"] + ".npy")))]

                    # Error calculations
                    # ====================================================================================================
                    e = err.get_errors(y_test, y_hat, anom, smoothed=False)
                    e_s = err.get_errors(y_test, y_hat, anom, smoothed=True)
                    errors_path = os.path.join("data", config.use_id, "errors", anom["chan_id"] + ".npy")
                    os.makedirs(os.path.dirname(errors_path), exist_ok=True)
                    np.save(errors_path, e)

                    errors_path = os.path.join("data", config.use_id, "smoothed_errors", anom["chan_id"] + ".npy")
                    os.makedirs(os.path.dirname(errors_path), exist_ok=True)
                    np.save(errors_path, e_s)

                    E_seq_test = eval(anom["anomaly_sequences"])
                    label_series = np.zeros(len(e_s), dtype=np.int32)
                    anom_path = os.path.join("data", config.use_id, "label_series", anom["chan_id"] + ".npy")
                    if os.path.exists(anom_path) and anom["chan_id"] == 'P-2':
                        label_series = np.load(anom_path)
                        # overlap two errors

                    for begin, end in E_seq_test:
                        for j in range(begin, min(end+1, len(e_s))):
                            label_series[j] = 1

                    os.makedirs(os.path.dirname(anom_path), exist_ok=True)
                    np.save(anom_path, label_series)



if __name__ == "__main__":
    config = Config("config.yaml")
    _id = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
    helpers.make_dirs(_id)  
    logger = helpers.setup_logging(config, _id)
    run(config, _id, logger)



    
