from ast import ExtSlice
from email.policy import default
import numpy as np
import os
import random
import torch
from sklearn.utils import class_weight

from src.data import create_dataloader, create_joint_dataloader
from src.scripts import *
from src.configs import *
from src.constants import *
from src.modules import Model,kaiming_init_weights

class UnitedNet:
    def __init__(self, save_path=None, device="cpu", technique=default_config):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.device = device
        self._create_model_for_technique(technique)

    def _set_device(self):
        self.model = self.model.to(device=self.device)
        self.model.device_in_use = self.device

    def _create_model_for_technique(self, technique):
        # current suported config: default_config, dbitseq_config, atacseq_config, dlpfc_config, patchseq_config,
        self._create_model_from_config(technique)

    def _create_model_from_config(self, config):
        self.model = Model(config)
        self.model.save_path = self.save_path
        self._set_device()

    def train(self, adatas_train, save_path=None,adatas_val=None,init_classify=False,verbose=False):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_path = save_path
        if str_label in adatas_train[0].obs.keys():
            labels = adatas_train[0].obs[str_label]
            self.model.class_weights = list(
                class_weight.compute_class_weight(
                    "balanced", classes=np.unique(labels), y=labels
                )
            )
        dataloader_train = create_dataloader(
            self.model,
            adatas_train,
            shuffle=True,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=True,
        )
        if adatas_val is None:
            adatas_val = adatas_train
        dataloader_test = create_dataloader(
            self.model,
            adatas_val,
            shuffle=False,
            batch_size=self.model.config[str_train_batch_size],
        )
        if init_classify:
            self.model.reset_classify()
            self._set_device()
        run_train(
            self.model, dataloader_train, dataloader_test,verbose=verbose
        )


    def finetune(self, adatas_finetune,save_path=None, adatas_val=None,init_classify=False,verbose=False):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_path = save_path
        dataloader_finetune = create_dataloader(
            self.model,
            adatas_finetune,
            shuffle=True,
            batch_size=self.model.config[str_finetune_batch_size],
        )
        if adatas_val is None:
            adatas_val = adatas_finetune
        dataloader_test = create_dataloader(
            self.model,
            adatas_val,
            shuffle=False,
            batch_size=self.model.config[str_finetune_batch_size],
        )
        if init_classify:
            self.model.reset_classify()
            self._set_device()
        run_finetune(
            self.model, dataloader_finetune, dataloader_test,verbose=verbose
        )

    def transfer(self, adatas_train, adatas_transfer,init_classify=False,save_path=None, adatas_val=None,verbose=False):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_path = save_path
        dataloader_train = create_dataloader(
            self.model,
            adatas_train,
            shuffle=True,
            batch_size=self.model.config[str_transfer_batch_size],
        )
        dataloader_train_and_transfer = create_joint_dataloader(
            self.model,
            adatas_train,
            adatas_transfer,
            shuffle=True,
            batch_size=self.model.config[str_transfer_batch_size],
        )
        if adatas_val is None:
            adatas_val = adatas_train
        dataloader_test = create_dataloader(self.model, adatas_val, shuffle=False)
        if init_classify:
            self.model.reset_classify()
            self._set_device()
        run_transfer(
            self.model,
            dataloader_train,
            dataloader_train_and_transfer,
            dataloader_test,
            verbose = verbose
        )

    def evaluate(self, adatas,give_losses=False,stage='train'):
        dataloader = create_dataloader(self.model, adatas, shuffle=False,)
        return run_evaluate(self.model, dataloader,give_losses=give_losses,stage=stage)

    def infer(self, adatas):
        dataloader = create_dataloader(self.model, adatas, shuffle=False,)
        return run_infer(self.model, dataloader)

    def predict(self, adatas):
        dataloader = create_dataloader(self.model, adatas, shuffle=False,)
        return run_predict(self.model, dataloader)

    def predict_label(self, adatas):
        dataloader = create_dataloader(self.model, adatas, shuffle=False, )
        return run_predict_label(self.model, dataloader)

    def load_model(self, path, device='cuda:0'):
        self.model = torch.load(path,map_location=torch.device(device))

