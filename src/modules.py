from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from src.constants import *

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_sizes = [config[str_input]] + config[str_hiddens]
        out_sizes = config[str_hiddens] + [config[str_output]]
        layers = []
        for in_size, out_size, use_bias, dropout, use_bn, use_ln, activation in zip(
            in_sizes,
            out_sizes,
            config[str_use_biases],
            config[str_dropouts],
            config[str_use_batch_norms],
            config[str_use_layer_norms],
            config[str_activations],
        ):
            layers.append(nn.Linear(in_size, out_size, use_bias))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_size))
            if use_ln:
                layers.append(nn.LayerNorm(out_size))
            if activation is not None:
                if activation == str_relu:
                    layers.append(nn.ReLU())
                elif activation == str_sigmoid:
                    layers.append(nn.Sigmoid())
                elif activation == str_softmax:
                    layers.append(nn.Softmax())
                elif activation == str_tanh:
                    layers.append(nn.Tanh())
                elif activation.startswith("leaky_relu"):
                    neg_slope = float(activation.split(":")[1])
                    layers.append(nn.LeakyReLU(neg_slope))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class WeightedMeanFuser(nn.Module):
    def __init__(self, n_modality):
        super().__init__()
        self.weights = nn.Parameter(
            torch.full((n_modality,),1/n_modality), requires_grad=True
        )

    def forward(self, latents):
        weights = F.softmax(self.weights, dim=0)
        weighted_latents = torch.sum(weights[None, :] * torch.stack(latents, dim=-1), dim=-1)
        return weighted_latents

class WeightedMeanFeatureFuser(nn.Module):
    def __init__(self, n_modality,l_dim):
        super().__init__()
        self.weights = nn.Parameter(
            torch.full((l_dim,n_modality),1/n_modality), requires_grad=True
        )

    def forward(self, latents):
        weights = nn.functional.softmax(self.weights, dim=-1)
        weighted_latents = torch.sum(weights[None, :] * torch.stack(latents, dim=-1), dim=-1)
        return weighted_latents

def kaiming_init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.translations = None
        self.input_dims = None
        self.config = config
        self.label_encoder = preprocessing.LabelEncoder()
        self.n_head = config[str_n_head]
        self.n_modality = len(config[str_encoders])
        self.noise_level = config[str_noise]
        self.register_buffer(str_best_head, torch.tensor(0, dtype=torch.long))
        self.potential_best_head = []
        self.register_buffer("head_flag", torch.tensor(0, dtype=torch.long))
        self.encoders = nn.ModuleList(
            [MLP(encoder) for encoder in config[str_encoders]]
        )
        self.decoders = nn.ModuleList(
            [MLP(decoder) for decoder in config[str_decoders]]
        )
        self.discriminators = nn.ModuleList(
            [MLP(discriminator) for discriminator in config[str_discriminators]]
        )
        self.fusers = nn.ModuleList(
            [WeightedMeanFeatureFuser(self.n_modality,config[str_encoders][0]["output"]) if config[str_fuser_type]=="WeightedFeatureMean" else WeightedMeanFuser(self.n_modality) for _ in range(self.n_head)]
        )
        self.latent_projector = MLP(config[str_latent_projector]) if config[str_latent_projector]!=None else nn.Identity()
        self.projectors = nn.ModuleList(
            [MLP(config[str_projectors]) for _ in range(self.n_head)]
        )
        self.clusters = nn.ModuleList(
            [MLP(config[str_clusters]) for _ in range(self.n_head)]
        )
        self.prob_layer = torch.nn.Softmax(dim=1)
        self.apply(kaiming_init_weights)
        self.train()

    def add_noise(self, inputs, levels, device):
        noised_input = []
        for input, level in zip(inputs, levels):
            shape = input.shape
            m_ = 0
            v_ = torch.var(input).detach() * level
            if v_ > 0:
                noise = torch.normal(m_, v_, size=shape).to(device=device)
                noised_input.append(input + noise)
            else:
                noised_input.append(input)
        return noised_input

    def impute_check(self,orig_modality):
        self.input_dims = [encoder["input"] for encoder in self.config[str_encoders]]
        if type(orig_modality) is not list:
            checked_modalities = []
            for sd in self.input_dims:
                if orig_modality.shape[1] == sd:
                    checked_modalities.append(torch.tensor(orig_modality))
                else:
                    checked_modalities.append(torch.zeros([orig_modality.shape[0], sd]))
        else:
            assert len(orig_modality) == self.n_modality, "please give either full list of all modalities or a single modality"
            checked_modalities = orig_modality
        return checked_modalities

    def forward(self, modalities, labels=None):
        modalities = self.impute_check(modalities)
        modalities = [
            modality.to(device=self.device_in_use) for modality in modalities
        ]

        if self.noise_level!=None:
            self.modalities = self.add_noise(inputs=modalities,levels=self.noise_level,device=self.device_in_use)

        self.labels = (
            labels.to(device=self.device_in_use) if labels is not None else None
        )

        self.latents = [
            encoder(modality)
            for (encoder, modality) in zip(self.encoders, self.modalities)
        ]

        with torch.no_grad():
            for pt_i in range(self.n_head):
                w = getattr(self.clusters[pt_i], "layers")[0].weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                getattr(self.clusters[pt_i], "layers")[0].weight.copy_(w)

        self.latent_projection = self.latent_projector(torch.cat(self.latents, dim=0))

        self.translations = [
            [decoder(latent) for latent in self.latents] for decoder in self.decoders
        ]

        self.discriminator_real_outputs = [
            discriminator(modality)
            for (discriminator, modality) in zip(self.discriminators, self.modalities)
        ]

        self.discriminator_fake_outputs = [
            discriminator(self.translations[i][i].detach())
            for i, discriminator in enumerate(self.discriminators)
        ]

        self.generator_outputs = [
            discriminator(self.translations[i][i])
            for i, discriminator in enumerate(self.discriminators)
        ]

        self.fused_latents = [fuser(self.latents) for fuser in self.fusers]

        self.hiddens = [
            projector(fused_latent)
            for (projector, fused_latent) in zip(self.projectors, self.fused_latents)
        ]

        self.cluster_outputs = [
            cluster(hidden) for (cluster, hidden) in zip(self.clusters, self.hiddens)
        ]

        self.predictions = [
            torch.argmax(self.prob_layer(cluster_outputs), axis=1)
            for cluster_outputs in self.cluster_outputs
        ]

        return (
            [
                [trans_lb if self.training else trans_lb.cpu().numpy() for trans_lb in trans_la] for
                trans_la in self.translations
            ],
            self.predictions[self.best_head] if self.training else self.predictions[self.best_head].cpu().numpy(),
            self.fused_latents[self.best_head] if self.training else self.fused_latents[self.best_head].cpu().numpy(),
        )

    def save_model(self, filename):
        if self.save_path is not None:
            self.modalities = None
            self.labels = None
            path = f"{self.save_path}/{filename}.pt"
            torch.save(self, path)

    def reset_classify(self):
        self.fusers = nn.ModuleList(
            [WeightedMeanFeatureFuser(self.n_modality,self.config[str_encoders][0]["output"]) if self.config[str_fuser_type]=="WeightedFeatureMean" else WeightedMeanFuser(self.n_modality) for _ in range(self.n_head)]
        )
        self.latent_projector = MLP(self.config[str_latent_projector]) if self.config[str_latent_projector]!=None else nn.Identity()
        self.projectors = nn.ModuleList(
            [MLP(self.config[str_projectors]) for _ in range(self.n_head)]
        )
        self.clusters = nn.ModuleList(
            [MLP(self.config[str_clusters]) for _ in range(self.n_head)]
        )
        self.prob_layer = torch.nn.Softmax(dim=1)


class submodel_trans(torch.nn.Module):
    def __init__(self, bigmodel, enc_dec_id):
        super(submodel_trans, self).__init__()
        self.encoder = bigmodel.encoders[enc_dec_id[0]]
        self.decoder = bigmodel.decoders[enc_dec_id[1]]

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class submodel_clus(torch.nn.Module):
    def __init__(self, bigmodel):
        super(submodel_clus, self).__init__()
        self.encoders = bigmodel.encoders
        self.fusers = bigmodel.fusers[bigmodel.best_head].weights
        self.projectors = bigmodel.projectors[bigmodel.best_head]
        self.n_head = bigmodel.n_head
        self.clusters = bigmodel.clusters[bigmodel.best_head]
        self.prob_layer = bigmodel.prob_layer
        self.best_head = bigmodel.best_head

    def fusing(self, fuser, latents):
        weights = nn.functional.softmax(fuser, dim=-1)
        weighted_latents = torch.sum(weights[None, :] * torch.stack(latents, dim=-1), dim=-1)
        return weighted_latents

    def forward(self, *modalities):
        self.latents = [
            encoder(modality)
            for (encoder, modality) in zip(self.encoders, modalities)
        ]

        self.fused_latents = self.fusing(self.fusers, self.latents)

        self.hiddens = self.projectors(self.fused_latents)

        with torch.no_grad():
            w = getattr(self.clusters, "layers")[0].weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            getattr(self.clusters, "layers")[0].weight.copy_(w)

        self.cluster_outputs = self.clusters(self.hiddens)

        self.predictions = self.prob_layer(self.cluster_outputs)

        return self.predictions

