from itertools import chain
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, r2_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from tqdm import tqdm

from src.data import anndata_from_outputs, create_dataloader
from src.loss import *
from src.constants import *
from tabulate import tabulate
import torch
import random


def sum_value_lists(list0, list1):
    if len(list0) == 0:
        return list1
    if len(list1) == 0:
        return list0
    elif len(list0) != len(list1):
        raise Exception("Please sum value lists of the same length.")
    else:
        combined_list = []
        for value0, value1 in zip(list0, list1):
            combined_list.append(value0 + value1)
    return combined_list


def amplify_value_dictionary_by_sample_size(dictionary, sample_size):
    amplified_dictionary = {}
    for key in dictionary:
        amplified_dictionary[key] = dictionary[key] * sample_size
    return amplified_dictionary


def average_dictionary_values_by_sample_size(dictionary, sample_size):
    if sample_size < 1:
        raise Exception("Please use positive count to average dictionary values.")
    for key in dictionary:
        dictionary[key] /= sample_size
    return dictionary


def sum_value_dictionaries(dictionary0, dictionary1):
    if not dictionary0:
        return dictionary1
    elif not dictionary1:
        return dictionary0

    combined_dictionary = {}
    for key in set(dictionary0.keys()).union(set(dictionary1.keys())):
        combined_dictionary[key] = dictionary0.get(key, 0) + dictionary1.get(key, 0)
    return combined_dictionary


def inplace_combine_tensor_lists(lists, new_list):
    """\
    In place add a new (nested) tensor list to current collections.
    This operation will move all concerned tensors to CPU.
    """
    if len(lists) == 0:
        for new_l in new_list:
            if isinstance(new_l, list):
                l = []
                inplace_combine_tensor_lists(l, new_l)
                lists.append(l)
            else:
                lists.append([new_l if type(new_l) == np.ndarray else new_l.detach().cpu()])
    else:
        for l, new_l in zip(lists, new_list):
            if isinstance(new_l, list):
                inplace_combine_tensor_lists(l, new_l)
            else:
                l.append(new_l if type(new_l) == np.ndarray else new_l.detach().cpu())


def concat_tensor_lists(lists):
    new_lists = []
    for l in lists:
        if len(l) == 0:
            raise Exception("Cannot concatenate empty tensor list.")
        if isinstance(l[0], list):
            new_lists.append(concat_tensor_lists(l))
        else:
            new_lists.append(np.concatenate(l,axis=0) if type(l[0])==np.ndarray else torch.cat(l, dim=0))
    return new_lists


class Schedule:
    def __init__(self, name, model, method, loss_weight, clas_flag=False):
        self.name = name
        self.best_loss = np.inf
        self.best_loss_term = None
        if name == str_classification:
            self.parameters = chain.from_iterable(
                [
                    model.encoders.parameters(),
                    model.fusers.parameters(),
                    model.latent_projector.parameters(),
                    model.projectors.parameters(),
                    model.clusters.parameters(),
                ]
            )
            self.optimizer = optim.Adam(self.parameters, lr=model.config[str_lr], )
            self.losses = [CrossEntropyLoss(model, loss_weight)]
            self.best_loss_term = str_cross_entropy_loss
        elif name == str_clustering:
            self.parameters = chain.from_iterable(
                [
                    model.fusers.parameters(),
                    model.latent_projector.parameters(),
                    model.projectors.parameters(),
                    model.clusters.parameters(),
                ]
            )
            self.optimizer = optim.Adam(self.parameters, lr=model.config[str_lr], )
            self.losses = [
                SelfEntropyLoss(model, loss_weight),
                DDCLoss(model, loss_weight),
                ReconstructionLoss(model, loss_weight),
            ]
            self.best_loss_term = str_ddc_loss
        if name == str_translation:
            self.parameters = chain.from_iterable(
                [
                    model.encoders.parameters(),
                    model.decoders.parameters(),
                    model.latent_projector.parameters(),
                    model.discriminators.parameters(),
                ]
            )
            self.optimizer = optim.Adam(self.parameters, lr=model.config[str_lr], )
            if ((method == str_finetune) and clas_flag) or (method == str_transfer):
                self.losses = [
                    ContrastiveLoss(model, loss_weight),
                    ReconstructionLoss(model, loss_weight),
                    TranslationLoss(model, loss_weight),
                ]
            else:
                self.losses = [
                    ContrastiveLoss(model, loss_weight),
                    ReconstructionLoss(model, loss_weight),
                    TranslationLoss(model, loss_weight),
                    DiscriminatorLoss(model, loss_weight),
                    GeneratorLoss(model, loss_weight),
                ]

            self.best_loss_term = str_translation_loss


    def step(self, model, train_model):
        if train_model:
            self.optimizer.zero_grad()

        losses = {}
        accumulated_head_losses = []
        total_loss = 0
        for loss in self.losses:
            if loss.name in [str_self_entropy_loss, str_ddc_loss, str_cross_entropy_loss]:
                _, head_losses = loss(model)
                for hd, h_ls in enumerate(head_losses):
                    losses[f'{loss.name}_head_{hd}'] = h_ls
                    total_loss += losses[f'{loss.name}_head_{hd}']
            else:
                losses[loss.name], head_losses = loss(model)
                total_loss += losses[loss.name]

            if train_model and head_losses is not None:
                accumulated_head_losses = sum_value_lists(
                    accumulated_head_losses, head_losses
                )

        if train_model:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, 25)
            self.optimizer.step()
        return losses

    def check_and_save_best_model(self, model, losses, best_model_path, verbose=False):
        if self.best_loss_term is None:
            curr_loss = sum(losses.values())
        else:
            if self.best_loss_term in [str_self_entropy_loss, str_ddc_loss, str_cross_entropy_loss]:
                curr_loss = losses[f'{self.best_loss_term}_head_{model.best_head}']
            else:
                curr_loss = losses[self.best_loss_term]
        if curr_loss < self.best_loss:
            model.save_model(best_model_path)
            self.best_loss = curr_loss
            if verbose:
                print("\n")
                headers = ["Losses", "Value"]
                losses['best_head'] = model.best_head
                values = list(losses.items())
                print(tabulate(values, headers=headers))
                print(f"best model saved at {model.save_path}/{best_model_path}.pt", "\n")


def run_schedule(runner):
    def wrapper_run_schedule(
            model,
            dataloader,
            schedule=None,
            train_model=False,
            infer_model=False,
            best_model_path=None,
            give_losses=False,
            verbose=False
    ):
        if train_model and schedule is not None and schedule.name == str_classification:
            for _ in range(len(dataloader.dataset.modalities)*2):
                outputs = runner(
                    model,
                    dataloader,
                    schedule,
                    train_model,
                    infer_model,
                    best_model_path,
                    give_losses=give_losses,
                    verbose=verbose
                )
            return outputs
        else:
            return runner(
                model, dataloader, schedule, train_model, infer_model, best_model_path,
                give_losses=give_losses, verbose=verbose
            )

    return wrapper_run_schedule


@run_schedule
def run_through_dataloader(
        model,
        dataloader,
        schedule=None,
        train_model=False,
        infer_model=False,
        best_model_path=None,
        give_losses=False,
        verbose=False
):
    all_outputs = []
    all_losses = {}

    for modalities, labels in dataloader:
        outputs = model(modalities, labels)
        if schedule is not None:
            losses = schedule.step(model, train_model)
            losses = amplify_value_dictionary_by_sample_size(losses, len(labels))
            all_losses = sum_value_dictionaries(all_losses, losses)

        if infer_model:
            inplace_combine_tensor_lists(all_outputs, outputs)

    if all_losses:
        all_losses = average_dictionary_values_by_sample_size(
            all_losses, len(dataloader.dataset)
        )
        for ls_name in all_losses:
            if ('ddc' in ls_name) or ('cross_entropy' in ls_name):
                ls_name_hd = '_'.join(ls_name.split('_')[:-1])
                head_losses = {k: all_losses[k] for k in all_losses.keys() if ls_name_hd in k}
                current_best_head = torch.tensor(int(min(head_losses, key=head_losses.get).split('_')[-1]))
                if hasattr(model, 'potential_best_head'):
                    model.potential_best_head.append(current_best_head)
                model.best_head = current_best_head
                break

    if best_model_path is not None:
        if len(model.potential_best_head)>0:
            cur_bc_heads = torch.bincount(torch.tensor(model.potential_best_head))
            if any(cur_bc_heads >= model.config[str_train_epochs]//3):
                model.best_head = torch.argmax(cur_bc_heads)
        else:
            model.best_head = torch.tensor(0, dtype=torch.long)
        schedule.check_and_save_best_model(model, all_losses, best_model_path, verbose=verbose)
    if give_losses:
        assert len(all_losses.keys()) > 0, 'wrong losses, the losses are empty'
        return all_losses
    else:
        return concat_tensor_lists(all_outputs)


def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered

def evalaute_outputs(dataloader, outputs):
    if not isinstance(dataloader.sampler, D.SequentialSampler):
        raise Exception("Please only evaluate outputs with non-shuffling dataloader.")
    dataset = dataloader.dataset
    labels = dataset.labels
    labels = labels if type(labels)==np.ndarray else labels.numpy()
    translations_outputs, predictions, *_ = outputs
    predictions = predictions if type(predictions)==np.ndarray else predictions.numpy()

    r2s = [
        [
            r2_score(modality if type(modality)==np.ndarray else modality.numpy(), translation if type(translation)==np.ndarray else translation.numpy())
            for translation in translations
        ]
        for modality, translations in zip(dataset.modalities, translations_outputs)
    ]
    accuracy, conf_mat = ordered_cmat(labels,predictions)
    metrics = {
        "r2": np.array(r2s),
        "confusion": conf_mat,
        "acc": accuracy,
        "ari": adjusted_rand_score(labels, predictions),
        "nmi": normalized_mutual_info_score(
            labels, predictions, average_method="geometric"
        ),
    }
    return metrics


def get_schedules_by_task(task, model):
    schedules_by_task = {
        str_supervised_group_identification: [str_translation, str_classification, ],
        str_unsupervised_group_identification: [str_translation, str_clustering],
        str_cross_model_prediction_clas: [str_classification, str_translation],
        str_cross_model_prediction_clus: [str_clustering, str_translation],
        str_cross_model_prediction: [str_translation],
        str_supervised_group_identigy_only:[str_classification],
        str_supervised_all:[str_all_supervised],


    }
    return schedules_by_task[task]


def run_train(model, dataloader_train, dataloader_val, verbose=False):
    print("training")
    task = model.config[str_train_task]
    loss_weight = model.config[str_train_loss_weight]
    clas_flag = ("clas" in task) or ("supervised" in task)
    schedules = [
        Schedule(schedule, model, str_train, loss_weight, clas_flag)
        for schedule in get_schedules_by_task(task, model)
    ]
    for epoch in tqdm(range(model.config[str_train_epochs])):
        epoch += 1
        model.cur_epoch = epoch
        model.train()
        for schedule in schedules:
            run_through_dataloader(model, dataloader_train, schedule, train_model=True)

        model.eval()
        with torch.no_grad():
            run_through_dataloader(
                model,
                dataloader_val,
                schedule,
                best_model_path=f"{str_train}_{str_best}",
                verbose=verbose
            )
        if epoch % model.config[str_checkpoint] == 0:
            model.save_model(f"{str_train}_epoch_{epoch}")
            if verbose:
                print(f"model saved at {model.save_path}/{str_train}_epoch_{epoch}.pt", "\n")

        if verbose:
            metrics = run_evaluate(model, dataloader_val)
            print("\n")
            headers = ["Metrics", "Value"]
            values = list(metrics.items())
            print(tabulate(values, headers=headers))


def run_finetune(model, dataloader_finetune, dataloader_val, verbose=False):
    print("finetuning")
    task = model.config[str_finetune_task]
    loss_weight = model.config[str_finetune_loss_weight]
    clas_flag = ("clas" in task) or ("supervised" in task)
    schedules = [
        Schedule(schedule, model, str_finetune, loss_weight, clas_flag)
        for schedule in get_schedules_by_task(task, model)
    ]
    for epoch in tqdm(range(model.config[str_finetune_epochs])):
        epoch += 1
        model.train()
        for schedule in schedules:
            run_through_dataloader(
                model, dataloader_finetune, schedule, train_model=True
            )

        model.eval()
        with torch.no_grad():
            run_through_dataloader(
                model,
                dataloader_val,
                schedule,
                best_model_path=f"{str_finetune}_{str_best}",
                verbose=verbose
            )
        if epoch % model.config[str_checkpoint] == 0:
            model.save_model(f"{str_finetune}_epoch_{epoch}")
            if verbose:
                print(f"model saved at {model.save_path}/{str_finetune}_epoch_{epoch}.pt", "\n")

        if verbose:
            metrics = run_evaluate(model, dataloader_val)
            print("\n")
            headers = ["Metrics", "Value"]
            values = list(metrics.items())
            print(tabulate(values, headers=headers))


def run_transfer(
        model, dataloader_train, dataloader_train_and_transfer, dataloader_val, verbose=False
):
    print("transferring")
    task = model.config[str_transfer_task]
    loss_weight = model.config[str_transfer_loss_weight]
    clas_flag = ("clas" in task) or ("supervised" in task)
    schedules = [
        Schedule(schedule, model, str_transfer, loss_weight, clas_flag)
        for schedule in get_schedules_by_task(task, model)
    ]
    for epoch in tqdm(range(model.config[str_transfer_epochs])):
        epoch += 1
        model.train()
        for schedule in schedules:
            if schedule.name in [str_classification, str_all_supervised]:
                run_through_dataloader(
                    model, dataloader_train, schedule, train_model=True
                )
            else:
                run_through_dataloader(
                    model, dataloader_train_and_transfer, schedule, train_model=True
                )

        model.eval()
        with torch.no_grad():
            run_through_dataloader(
                model,
                dataloader_val,
                schedule,
                best_model_path=f"{str_transfer}_{str_best}",
                verbose=verbose
            )

        if epoch % model.config[str_checkpoint] == 0:
            model.save_model(f"{str_transfer}_epoch_{epoch}")
            if verbose:
                print(f"model saved at {model.save_path}/{str_transfer}_epoch_{epoch}.pt", "\n")

        if verbose:
            metrics = run_evaluate(model, dataloader_val)
            print("\n")
            headers = ["Metrics", "Value"]
            values = list(metrics.items())
            print(tabulate(values, headers=headers))


def run_evaluate(model, dataloader, give_losses=False, stage='train'):
    model.eval()
    if give_losses:
        losses = {}
        task = model.config[globals()["str_%s_task" % (stage)]]
        loss_weight = model.config[globals()["str_%s_loss_weight" % (stage)]]
        clas_flag = ("clas" in task) or ("supervised" in task)
        schedules = [
            Schedule(schedule, model, globals()["str_%s" % (stage)], loss_weight, clas_flag)
            for schedule in get_schedules_by_task(task, model)
        ]
        with torch.no_grad():

            for ii, schedule in enumerate(schedules):
                loss = run_through_dataloader(model, dataloader, schedule, infer_model=True, give_losses=give_losses)
                losses[f'{task}_shedule{ii}'] = loss
        return losses
    else:
        with torch.no_grad():
            outputs = run_through_dataloader(model, dataloader, infer_model=True, give_losses=give_losses)
        return evalaute_outputs(dataloader, outputs)


def run_infer(model, dataloader):
    model.eval()
    with torch.no_grad():
        outputs = run_through_dataloader(model, dataloader, infer_model=True)
    return anndata_from_outputs(model, outputs)


def run_predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        outputs = run_through_dataloader(model, dataloader, infer_model=True)
    return outputs[0]


def run_predict_label(model, dataloader):
    model.eval()
    with torch.no_grad():
        outputs = run_through_dataloader(model, dataloader, infer_model=True)
    return outputs[1]

from sklearn.metrics import confusion_matrix
import copy

def assignmene_align(labels1, labels2):
    C_e_types = confusion_matrix(labels1, labels2)
    # Assign labels of clusters based on 'best match' with transcriptomic celltype label
    row_ind, col_ind = linear_sum_assignment(-C_e_types)
    order_2 = np.unique(labels2)[col_ind]
    labels2_matched = copy.deepcopy(labels2)
    for name, orig_name in zip(np.unique(labels2), order_2):
        ind = labels2 == orig_name
        labels2_matched[ind] = name
    return labels2_matched
