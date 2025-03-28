import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from DataLoaderBase import ArrayDataLoader, SplitTrain, SplitValid
from log import get_logger
from model import deepConvNet


def ResultsSaveDirection(Common_config, Client_config_orig):
    """
    Set the results saving direction path according to their setting parameters and save the arguments setting
    Args:
        Common_config: the common config
        TargetC_config: the config of the target client
        OtherC_config_orig: the config of other clients

    Returns:
        base_dir: str, the path direction
    """
    client_base_dir_list = []

    if Common_config["subject_wise"]:
        base_dir = "./Results/{}/{}Clients_{}Rounds_SubjectWise/".format(
            Common_config["server_aggregation"],
            Common_config["num_clients"],
            Common_config["rounds"]
        )
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        base_dir_client_weights = base_dir + "Client_weights/"
        if not os.path.exists(base_dir_client_weights):
            os.makedirs(base_dir_client_weights, exist_ok=True)

        base_dir_logger = base_dir + "Logger/{}_{}fold/".format(
            Common_config["num_folds"][0],
            Common_config["num_folds"][1],
        )
        if not os.path.exists(base_dir_logger):
            os.makedirs(base_dir_logger, exist_ok=True)

        base_dir_temp_local_models = base_dir + "temp_local_models/{}_{}fold/".format(
            Common_config["num_folds"][0],
            Common_config["num_folds"][1],
        )
        if not os.path.exists(base_dir_temp_local_models):
            os.makedirs(base_dir_temp_local_models, exist_ok=True)

        for client_config in Client_config_orig:
            temp_base_dir = base_dir + "{}/{}_{}fold/".format(
                client_config["name"],
                Common_config["num_folds"][0],
                Common_config["num_folds"][1],
            )
            if not os.path.exists(temp_base_dir):
                os.makedirs(temp_base_dir, exist_ok=True)
            client_base_dir_list.append(temp_base_dir)
    else:
        base_dir = "./Results/{}/{}Clients_{}Rounds_TrialWise/".format(
            Common_config["server_aggregation"],
            Common_config["num_clients"],
            Common_config["rounds"]
        )
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        base_dir_client_weights = base_dir + "Client_weights/"
        if not os.path.exists(base_dir_client_weights):
            os.makedirs(base_dir_client_weights, exist_ok=True)

        base_dir_logger = base_dir + "Logger/{}_{}fold/".format(
            Common_config["num_folds"][0],
            Common_config["num_folds"][1],
        )
        if not os.path.exists(base_dir_logger):
            os.makedirs(base_dir_logger, exist_ok=True)

        base_dir_temp_local_models = base_dir + "temp_local_models/{}_{}fold/".format(
            Common_config["num_folds"][0],
            Common_config["num_folds"][1],
        )
        if not os.path.exists(base_dir_temp_local_models):
            os.makedirs(base_dir_temp_local_models, exist_ok=True)

        for client_config in Client_config_orig:
            temp_base_dir = base_dir + "Client_{}/{}_{}fold/".format(
                client_config["name"],
                Common_config["num_folds"][0],
                Common_config["num_folds"][1],
            )
            if not os.path.exists(temp_base_dir):
                os.makedirs(temp_base_dir, exist_ok=True)
            client_base_dir_list.append(temp_base_dir)

    # Save the arguments settings
    setting_file = os.path.join(base_dir, "Configs_setting.txt")
    if not os.path.exists(setting_file):
        argument_file = open(setting_file, "w")
        argument_file.write("Common settings:\n")
        argument_file.write(str(Common_config))
        argument_file.write("\n")
        argument_file.write("Clients' settings:\n")
        for item in Client_config_orig:
            argument_file.write(str(item))
            argument_file.write("\n")
        argument_file.close()

    return client_base_dir_list, base_dir_logger, base_dir_temp_local_models, base_dir_client_weights


def TargetClientInitialization(
    idx_fold, test_index, Common_config, TargetC_config, shared_data_array
):
    """
    Initialize the local model, optimizer, learning rate scheduler and arguments for the target client inside the client
    Args:
        idx_fold: the index of the fold
        test_index: the index of the subjects selected as testing set for the current fold
        Common_config: the config of the FL system
        TargetC_config: the config of the target client
        shared_data_array: the shared data array for the target client

    Returns:
        target_train_dataloader: the target client's training dataloader
        target_valid_dataloader: the target client's validation dataloader
        target_test_dataloader: the target client's testing dataloader
        target_model: local model
        target_optimizer: local optimizer
        target_lr_scheduler: local learning rate scheduler
        w_glob_keys: get the global layers' names
    """

    # get_logger().info("The target client is Client {}".format(TargetC_config["name"]))

    # load the corresponding fold data for the target client
    target_train_dataloader = DataLoader(
        SplitTrain(shared_data_array, test_index, Common_config),
        batch_size=TargetC_config["batch_size"],
        shuffle=True,
    )
    target_valid_dataloader = DataLoader(
        SplitValid(shared_data_array, test_index, Common_config),
        batch_size=TargetC_config["batch_size"],
        shuffle=True,
    )
    target_test_dataloader = [DataLoader(
        ArrayDataLoader(shared_data_array[idx]),
        batch_size=TargetC_config["test_batch_size"],
        shuffle=False,
    ) for idx in test_index]

    # Initialize the local model
    target_model = deepConvNet(
        nChan=TargetC_config["nChan"],
        nTime=TargetC_config["nTime"],
        poolSize=TargetC_config["poolSize"],
        localKernalSize=TargetC_config["localKernalSize"],
    )
    target_model.train()
    get_logger().info(
        "Initialize the model in Client {}".format(TargetC_config["name"])
    )
    if idx_fold == 0:
        get_logger().info(target_model)

    # optimizer for the target client
    if TargetC_config["optim_type"] == "adam":
        target_optimizer = torch.optim.Adam(
            target_model.parameters(),
            lr=TargetC_config["lr"],
            weight_decay=0,
            amsgrad=False,
        )
    if TargetC_config["optim_type"] == "sgd":
        target_optimizer = torch.optim.SGD(
            target_model.parameters(),
            lr=TargetC_config["lr"],
            momentum=TargetC_config["momentum"],
        )
    if TargetC_config["optim_type"] == "adamW":
        target_optimizer = torch.optim.AdamW(
            target_model.parameters(),
            lr=TargetC_config["lr"],
            weight_decay=0.5 * 0.001,
            amsgrad=False,
        )

    # setting the lr_scheduler
    n_updates_per_epoch = 0
    for _ in target_train_dataloader:
        n_updates_per_epoch += 1
    T_max = n_updates_per_epoch * TargetC_config["local_ep"] * Common_config["rounds"]
    assert target_optimizer is not None
    target_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        target_optimizer, T_max
    )

    # get the first num_layers_keep layers as the local layers, the rest as global layers
    w_glob_keys = target_model.weight_keys[Common_config["num_layers_keep"]:]
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    return (
        target_train_dataloader,
        target_valid_dataloader,
        target_test_dataloader,
        target_model,
        target_optimizer,
        target_lr_scheduler,
        w_glob_keys,
    )



def SaveCurrentFoldRecords(idx_fold, target_client_valid_records, base_dir):
    """
    Save the validation results in the target client and plot the valid loss in the current fold
    Args:
        idx_fold: index of the current fold
        target_client_valid_records: validation and testing loss /accuracy results in the target client of the current fold
        base_dir: base path direction

    Returns:
        fold_save_path: the path direction of current fold
    """

    # set the current fold results saving path
    fold_save_path = os.path.join(base_dir, "Fold{}".format(idx_fold + 1))
    if not os.path.exists(fold_save_path):
        os.makedirs(fold_save_path, exist_ok=True)

    # save the training process records for the target client
    results_columns = [
        "rounds",
        "target_client_valid_acc",
        "target_client_valid_loss",
        "target_client_test_acc",
        "target_client_test_loss",
        "time cost",
    ]
    results_target_client_save = np.array(target_client_valid_records)
    results_target_client_save = pd.DataFrame(
        results_target_client_save, columns=results_columns
    )
    results_save_path = os.path.join(fold_save_path, "results_target_client.csv")
    results_target_client_save.to_csv(results_save_path, index=False)

    # build the folder that contains the validation loss figures
    figures_save_path = os.path.join(base_dir, "Valid_loss_figures")
    if not os.path.exists(figures_save_path):
        os.makedirs(figures_save_path, exist_ok=True)
    # plot the valid loss figure for the target client
    valid_loss = results_target_client_save["target_client_valid_loss"]
    rounds = results_target_client_save["rounds"]
    plt.plot(rounds, valid_loss)
    plt.xlabel("Rounds")
    plt.ylabel("Valid Loss")
    plt.title("Fold {}".format(idx_fold))
    fig_name = os.path.join(figures_save_path, "Fold{}.png".format(idx_fold))
    plt.savefig(fig_name)
    plt.close()

    return fold_save_path





