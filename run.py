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

import more_itertools as mit
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tqdm import trange, tqdm

from telemanom._globals import Config
import telemanom.errors as err
import telemanom.helpers as helpers
import telemanom.modeling as models

def load_data(
        anom_labels,
        anom_es,
        anom_paths,
        chan_id,
        space_id):
    anom_es = [torch.tensor(v, dtype=torch.float32) for v in anom_es]
    anom_paths = [torch.tensor(v, dtype=torch.float32) for v in anom_paths]
    data = []
    for i, (label, space) in enumerate(anom_labels):
        if space_id != space:
            if anom_es[i].shape[0] < anom_paths[i].shape[0]:
                data.append((anom_es[i].reshape(-1, 1),
                anom_paths[i][config.l_s:-config.n_predictions].reshape(-1, 1)))
            else:
                data.append((anom_es[i].reshape(-1, 1),
                anom_paths[i].reshape(-1, 1)))

    val_size = max(2, int(len(data)*0.1))
    train_size = len(data) - val_size
    train_set, val_set = random_split(data, [train_size, val_size])
    loader_trn = DataLoader(
                    train_set, batch_size=1, shuffle=True)
    loader_val = DataLoader(
                    val_set, batch_size=1, shuffle=True)
    return loader_trn, loader_val

class BasicGenerator(nn.Module):
    def __init__(self, latent_size=4, input_size=config.window_size, output_size=1):
        super().__init__()

        def block(num_inputs, num_outputs):
            return [nn.Linear(num_inputs, num_outputs), nn.ReLU()]

        self.model = nn.Sequential(*block(latent_size + input_size, 20),
                                   *block(20, 40), 
                                   nn.Linear(40, output_size))
        self.linear = nn.Linear(latent_size + input_size, output_size)

    def forward(self, z, c):
        inputs = torch.cat([z, c], dim=-1)
        return self.model(inputs) + self.linear(inputs)


class ScaleDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 40),
            nn.LeakyReLU(0.2),
            nn.Linear(40, 1))
        self.linear = nn.Linear(input_size, 1)

    def forward(self, inputs):
        return self.model(inputs) + self.linear(inputs)

class GAN(nn.Module):
    def __init__(self, latent_size, input_size=config.window_size, output_size=1):
        super().__init__()
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.criteria = torch.nn.MSELoss()
        self.criteria2 = torch.nn.BCEWithLogitsLoss()
        self.gen = BasicGenerator(
            latent_size=latent_size,
            input_size=input_size,
            output_size=output_size)
        self.disc = ScaleDiscriminator(input_size=input_size+output_size)

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def train(self, dataloader, lr, beta1, beta2, num_epochs):
        g_optim = torch.optim.Adam(
            self.gen.parameters(), lr=lr, betas=(beta1, beta2))
        d_optim = torch.optim.Adam(
            self.disc.parameters(), lr=lr, betas=(beta1, beta2))
        self.gen.train()
        self.disc.train()
        device = list(self.gen.parameters())[0].device
        for epoch in trange(num_epochs):
            t = tqdm(dataloader)
            for i, batch_data in enumerate(t):
                batch_data = batch_data.to(device)

                real_batch_size = batch_data.shape[0]
                zeros = torch.zeros(
                        real_batch_size,
                        1,
                        dtype=torch.float,
                        device=device)
                ones = torch.ones(
                        real_batch_size,
                        1,
                        dtype=torch.float,
                        device=device)

                c, y = torch.split(
                    batch_data, [4, 1], dim=-1)

                # generator
                g_optim.zero_grad()
                z = torch.tensor(
                        np.random.normal(0, 1,
                                         (real_batch_size, self.latent_size)),
                        dtype=torch.float,
                        device=device)
                    

                g_half = self.gen(z, c)
                fake = torch.cat([c, g_half], dim=-1)

                g_loss = self.criteria(self.disc(fake), y)
                # g_loss = self.criteria2(self.disc(fake), ones)

                g_loss.backward()
                g_optim.step()

                # discriminator
                d_optim.zero_grad()

                d_real_loss = self.criteria(self.disc(batch_data), y)
                d_fake_loss = 0.5*self.criteria(self.disc(fake.detach()), y)
                d_g_loss = (d_real_loss - d_fake_loss) / 2
                # d_real_loss = self.criteria2(self.disc(batch_data), ones)
                # d_fake_loss = self.criteria2(self.disc(fake.detach()), zeros)
                # d_g_loss = (d_real_loss + d_fake_loss) / 2

                d_g_loss.backward()
                d_optim.step()

                t.set_postfix(
                    g_sample=list(fake[0].data.cpu().numpy()),
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader)),
                    d_g_loss=d_g_loss.item(),
                    d_real_loss=d_real_loss.item(),
                    d_fake_loss=d_fake_loss.item(),
                    g_loss=g_loss.item())

class BasicRNN(nn.Module):
    def __init__(self, latent_size=10, input_size=1, output_size=1):
        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.latent_size = latent_size

        self.rnn_layer = nn.LSTM(self.input_size, self.latent_size, bidirectional=True, batch_first=True, num_layers=2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.latent_size*2, self.output_size), )
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.device = torch.device('cuda')

    def forward(self, inputs):
        outputs, hidden = self.rnn_layer(inputs)
        # outputs: [seq_len, batch, num_directions * hidden_size]

        outputs = self.fc_layer(outputs)
        # [seq_len, batch, output_size]

        return outputs

    def eval_dataset(self,
            dataloader):

        self.eval()
        total = 0
        for X, y in dataloader:
            X = X.to(self.device)
            y= y.to(self.device)
            total += y.size(1)
            outputs = self(X)
            predicts = torch.sigmoid(outputs) > 0.5
            total_correct = (predicts.to(torch.long) == y.to(torch.long)).sum().item()

        self.train()
        return total_correct / total

    def start_train(self,
              dataloader,
              val_dataloader,
              lr,
              beta1,
              beta2,
              num_epochs, 
              grad_clip=5.0,
              patience=5):
        optim = torch.optim.Adam(
                self.parameters(),
            lr=lr,
            betas=(beta1, beta2))
        self.train()
        hit_num = 0
        best_accuracy = curr_accuracy= 0
        best_epoch = 0
        for epoch in trange(num_epochs):
            t = tqdm(dataloader)

            for i, (X, y) in enumerate(t):
                X = X.to(self.device)
                y= y.to(self.device)
                optim.zero_grad()
                outputs = self(X)
                loss = self.criteria(outputs, y)
                loss.backward()
                clip_grad_norm_(self.parameters(),
                                grad_clip)
                optim.step()

                t.set_postfix(
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader)),
                    patience=patience,
                    loss=loss.item(),
                    best_accuracy=best_accuracy,
                    curr_accuracy=curr_accuracy,
                    best_epoch=best_epoch,
                )

            curr_accuracy = acc = self.eval_dataset(val_dataloader)
            if acc > best_accuracy:
                best_accuracy = acc
                best_epoch = epoch
                hit_num = 0
            else:
                hit_num += 1
                if hit_num > patience:
                    break


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

    anom_labels = []
    anom_es = []
    anom_paths = []
    with open("labeled_anomalies.csv", "rU") as f:
        reader = csv.DictReader(f)
        for i, anom in enumerate(reader):
            if reader.line_num >= 1:
                errors_path = os.path.join("data", config.use_id, "smoothed_errors", anom["chan_id"] + ".npy")
                anom_path = os.path.join("data", config.use_id, "anom_series", anom["chan_id"] + ".npy")
                anom_labels.append((anom["chan_id"], anom["spacecraft"]))
                anom_es.append(np.load(errors_path))
                anom_paths.append(np.load(anom_path))


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
                    y_hat = []
                    if config.predict:
                        model = models.get_model(anom, X_train, y_train, logger, train=config.train)
                        y_hat = models.predict_in_batches(y_test, X_test, model, anom)
                            
                    else:
                        y_hat = [float(x) for x in list(np.load(os.path.join("data", config.use_id, "y_hat", anom["chan_id"] + ".npy")))]

                    # Error calculations
                    # ====================================================================================================
                    errors_path = os.path.join("data", config.use_id, "errors", anom["chan_id"] + ".npy")
                    if config.errors or not os.path.exists(errors_path):
                        e = err.get_errors(y_test, y_hat, anom, smoothed=False)
                        os.makedirs(os.path.dirname(errors_path), exist_ok=True)
                        np.save(errors_path, e)
                    else:
                        e = np.load(errors_path)

                    errors_path = os.path.join("data", config.use_id, "smoothed_errors", anom["chan_id"] + ".npy")
                    if config.errors or not os.path.exists(errors_path):
                        e_s = err.get_errors(y_test, y_hat, anom, smoothed=True)
                        os.makedirs(os.path.dirname(errors_path), exist_ok=True)
                        np.save(errors_path, e_s)
                    else:
                        e_s = np.load(errors_path)

                    
                    anom_series = np.zeros(e_s.shape[0]+config.l_s +config.n_predictions, dtype=np.int32)
                    # l_s, e_s, n_predictions
                    E_seq_test = eval(anom["anomaly_sequences"])
                    for begin, end in E_seq_test:
                        for j in range(begin, end+1):
                            anom_series[j] = 1
                    anom_path = os.path.join("data", config.use_id, "anom_series", anom["chan_id"] + ".npy")
                    if not os.path.exists(anom_path):
                        os.makedirs(os.path.dirname(anom_path), exist_ok=True)
                        np.save(anom_path, anom_series)


                    anom["normalized_error"] = np.mean(e) / np.ptp(y_test)
                    logger.info("normalized prediction error: %s" %anom["normalized_error"])

                    # Error processing (batch)
                    # =========================
                    if config.transfer:
                        if config.transfer_space:
                            model_path = os.path.join("data", config.use_id, "space_model", anom["spacecraft"] + ".pth")
                        else:
                            model_path = os.path.join("data", config.use_id, "post_models", anom["chan_id"] + ".pth")
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        device = torch.device('cuda')
                        if not os.path.exists(model_path):
                            model = BasicRNN().to(device)
                            dataloader_trn, dataloader_val = load_data(
                                anom_labels,
                                anom_es,
                                anom_paths,
                                anom["chan_id"],
                                anom["spacecraft"])
                            model.start_train(
                                dataloader_trn,
                                dataloader_val,
                                lr=0.0001,
                                beta1=0.5,
                                beta2=0.999,
                                num_epochs=10) 
                            torch.save(model, model_path)
                        else:
                            model = torch.load(model_path)

                        predicts = torch.sigmoid(model(torch.tensor(e_s.reshape(-1, 1, 1), dtype=torch.float32).to(device)))
                        predicts = predicts.data.cpu().numpy().reshape(-1)
                        E_seq, E_seq_scores = err.process_errors(y_test, y_hat, predicts, anom, logger)
                        # print(' '.join(str(p) for p in predicts))
                        # i_anom = []
                        # for idx, p in enumerate(predicts):
                        #     if p > 0.5:
                        #         i_anom.append(idx + config.l_s)

                        # groups = [list(group) for group in mit.consecutive_groups(i_anom)]
                        # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
                        # E_seq = [(0, 9999999)]
                        # E_seq_scores = []
                        
                    else:
                        E_seq, E_seq_scores = err.process_errors(y_test, y_hat, e_s, anom, logger)
                    anom['scores'] = E_seq_scores

                    anom = err.evaluate_sequences(E_seq, anom)
                    anom["num_values"] = y_test.shape[0] + config.l_s + config.n_predictions

                    for key, value in stats.items():
                        stats[key] += anom[key]

                    helpers.anom_stats(stats, anom, logger)
                    writer.writerow(anom)

    helpers.final_stats(stats, logger)


if __name__ == "__main__":
    config = Config("config.yaml")
    _id = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
    helpers.make_dirs(_id)  
    logger = helpers.setup_logging(config, _id)
    run(config, _id, logger)



    
