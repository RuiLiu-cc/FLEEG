import copy
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from log import get_logger
from utils import (SaveCurrentFoldRecords, TargetClientInitialization)


def client(
    idx_client,
    idx_fold,
    test_index,
    base_dir,
    temp_local_model_path,
    Common_config,
    Client_config,
    shared_data_array,
    queue_network,
):
    """
    Entire local training, validation and testing in the target client
    Args:
        idx_client: the index of the client
        idx_fold: the index of the current training round
        test_index: the index of the subjects selected as validation set for the current fold.
        base_dir: the folder direction of the results
        temp_local_model_path: the folder direction of the temporal local models
        Common_config: the common config
        Client_config: the config of the client
        shared_data_array: the shared data array for the client
        queue_network: the entire queue networks
    Returns:
        client_best_test_acc
        client_best_test_loss
    """
    # Initialization
    (
        target_train_dataloader,
        target_valid_dataloader,
        target_test_dataloader,
        model,
        optimizer,
        lr_scheduler,
        w_glob_keys,
    ) = TargetClientInitialization(
        idx_fold, test_index, Common_config, Client_config, shared_data_array
    )

    # set the checking point path:
    if Common_config["save_checkingpoint"]:
        checkingpoint_save_path = os.path.join(base_dir, "CheckingPoint")
        if not os.path.exists(checkingpoint_save_path):
            os.makedirs(checkingpoint_save_path, exist_ok=True)
        checkingpoint_save_name = os.path.join(checkingpoint_save_path, "Client_{}.pt".format(Client_config["name"]))


    # Training round iteration
    target_client_valid_records = []
    best_round = -1
    target_client_best_val_loss = 100000
    optimal_round_val_acc = -1
    target_client_optimal_val_acc = 0

    # Send the global layers' key to the server
    if idx_client == 0:
        queue_network.send_to_server(w_glob_keys, idx_client)

    for idx_round in range(Common_config["rounds"]):
        time_cost = local_train(
            model, optimizer, lr_scheduler, target_train_dataloader, Client_config
        )
        get_logger().info(
            "Client: {} ({}) finishes round {} training in {:.2f}s.".format(
            Client_config["name"],
            idx_client,
            idx_round,
            time_cost
            )
        )

        model.cpu()
        model_state_dict = model.state_dict()
        if Common_config["server_aggregation"] == "ValidSetWeights":
            # send the entire model to the server process
            global_layers_model = model
        else:
            global_layers_model = {}
            for k in w_glob_keys:
                global_layers_model[k] = model_state_dict[k]
        # save the dictionary
        temp_local_model_filename = os.path.join(temp_local_model_path, "Client_{}.pt".format(Client_config["name"]))
        torch.save(global_layers_model, temp_local_model_filename)
        queue_network.send_to_server(temp_local_model_filename, idx_client)

        # Wait for the updated global layer from the server
        while True:
            # get the updated global layer from the server
            if queue_network.server_has_data(idx_client):
                global_layers_filename = queue_network.get_from_server(idx_client)
                updated_global_layers_model = torch.load(global_layers_filename, map_location=torch.device('cpu'))
                break
            time.sleep(0.5)

        # update the local model with updated global layers
        for k in w_glob_keys:
            model_state_dict[k] = updated_global_layers_model[k]
        model.load_state_dict(model_state_dict)

        # Valid the updated model on the validation set in the target client
        valid_loss, valid_acc = local_valid_test(
            model, Client_config, target_valid_dataloader
        )
        get_logger().info(
            "Client: {} ({}) Valid Acc: {:.2f}, Valid Loss: {:.6f} at round {}.".format(
                Client_config["name"], idx_client,
                valid_acc, valid_loss, idx_round
            )
        )
        target_client_valid_records.append(
            np.array([idx_round, valid_acc, valid_loss, None, None, time_cost])
        )

        # Check the valid loss and store the best model that give the best valid loss for the target client
        if valid_loss < target_client_best_val_loss:
            target_client_best_val_loss = valid_loss
            target_client_best_val_acc = valid_acc
            best_round = idx_round
            best_local_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)
            best_lr_scheduler = copy.deepcopy(lr_scheduler)


        # Check the valid acc and store the best model that give the best valid acc for the target client
        if valid_acc > target_client_optimal_val_acc:
            target_client_optimal_val_acc = valid_acc
            target_client_optimal_val_loss = valid_loss
            optimal_round_val_acc = idx_round
            optimal_local_model = copy.deepcopy(model)
            optimal_optimizer = copy.deepcopy(optimizer)
            optimal_lr_scheduler = copy.deepcopy(lr_scheduler)


        # save the checking point
        if Common_config["save_checkingpoint"]:
            if idx_round % Common_config["checkingpoint_step"]:
                torch.save({
                    'idx_fold': idx_fold,
                    'idx_round': idx_round,
                    'model': model,
                    'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler,
                    'target_client_valid_records': target_client_valid_records,
                    'best_round': best_round,
                    'target_client_best_val_loss': target_client_best_val_loss,
                    'best_local_model': best_local_model,
                    'best_optimizer': best_optimizer,
                    'best_lr_scheduler': best_lr_scheduler,
                }, checkingpoint_save_name)

    get_logger().info("----------------------------------")
    get_logger().info("Results: {} ({}) get best valid loss results at {} rounds:".format(
            Client_config["name"],
            idx_client,
            best_round
        )
    )
    get_logger().info(
        "Results: {} ({}) Valid Acc: {:.2f}, Valid Loss: {:.6f}".format(
            Client_config["name"], idx_client,
            target_client_best_val_acc, target_client_best_val_loss
        )
    )

    # test the best model with optimal valid loss on the testing data set for the target clients
    target_client_best_test_loss_list = []
    target_client_best_test_acc_list = []
    for (sub_idx, one_target_test_dataloader) in zip(test_index, target_test_dataloader):
        target_client_best_test_loss, target_client_best_test_acc = local_valid_test(
            best_local_model, Client_config, one_target_test_dataloader
        )
        get_logger().info(
            "Results: {} ({}) its sub {}: Test Acc: {:.2f}, Test Loss: {:.6f}".format(
                Client_config["name"], idx_client,
                sub_idx, target_client_best_test_acc, target_client_best_test_loss
            )
        )
        target_client_best_test_loss_list.append(target_client_best_test_loss)
        target_client_best_test_acc_list.append(target_client_best_test_acc)

    get_logger().info(
        "Results: {} ({}) get best valid acc results at {} rounds:".format(
            Client_config["name"],
            idx_client,
            optimal_round_val_acc
        )
    )
    get_logger().info(
        "Results: {} ({}) Valid Acc: {:.2f}, Valid Loss: {:.6f}".format(
            Client_config["name"], idx_client,
            target_client_optimal_val_acc, target_client_optimal_val_loss
        )
    )
    # test the best model with optimial valid acc on the testing data set for the target clients
    target_client_optimal_test_loss_list = []
    target_client_optimal_test_acc_list = []
    for (sub_idx, one_target_test_dataloader) in zip(test_index, target_test_dataloader):
        target_client_optimal_test_loss, target_client_optimal_test_acc = local_valid_test(
            optimal_local_model, Client_config, one_target_test_dataloader
        )
        get_logger().info(
            "Results: {} ({}) its sub {}: Test Acc: {:.2f}, Test Loss: {:.6f}".format(
                Client_config["name"], idx_client,
                sub_idx, target_client_optimal_test_acc, target_client_optimal_test_loss
            )
        )
        target_client_optimal_test_loss_list.append(target_client_optimal_test_loss)
        target_client_optimal_test_acc_list.append(target_client_optimal_test_acc)

    get_logger().info("----------------------------------")

    # save the validation and testing results in the current fold
    target_client_valid_records.append(
        np.array(
            [
                best_round,
                target_client_best_val_acc,
                target_client_best_val_loss,
                None,
                None,
                None
            ]
        )
    )
    target_client_valid_records.append(
        np.array(
            [
                optimal_round_val_acc,
                target_client_optimal_val_acc,
                target_client_optimal_val_loss,
                None,
                None,
                None
            ]
        )
    )
    fold_save_path = SaveCurrentFoldRecords(
        idx_fold, target_client_valid_records, base_dir
    )

    # save the best model for the client
    model_save_path = os.path.join(fold_save_path, "target_client_best_model.pt")
    torch.save(
        {
            "model_state_dict": best_local_model.state_dict(),
            "optimizer_state_dict": best_optimizer.state_dict(),
            "lr_scheduler_state_dict": best_lr_scheduler.state_dict(),
        },
        model_save_path,
    )
    model_save_path = os.path.join(fold_save_path, "target_client_optimal_model.pt")
    torch.save(
        {
            "model_state_dict": optimal_local_model.state_dict(),
            "optimizer_state_dict": optimal_optimizer.state_dict(),
            "lr_scheduler_state_dict": optimal_lr_scheduler.state_dict(),
        },
        model_save_path,
    )

    return target_client_best_test_acc_list, target_client_best_test_loss_list, target_client_optimal_test_acc_list, target_client_optimal_test_loss_list




# define the local training function for one client
def local_train(local_model, optimizer, lr_scheduler, data_loader, client_config):
    """
    Train the local model in the client within a certain epoch
    Args:
        local_model: local model
        optimizer: local optimizer
        lr_scheduler: local lr scheuler
        data_loader: local dataloader
        client_config: local config
        w_glob_keys: global layers' keys

    Returns:
        updated_local_model: the updated local models
    """

    loss_func = nn.NLLLoss()
    device = (
        str("cuda:" + client_config["device"]) if torch.cuda.is_available() else "cpu"
    )
    client_data_loader = data_loader
    local_model.to(device)
    local_model.train()
    start_time = time.time()


    for _ in range(client_config["local_ep"]):
        for data, labels in client_data_loader:
            data, labels = data.to(device), labels.to(device)
            local_model.zero_grad()
            log_probs = local_model(data)

            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    time_cost = time.time() - start_time

    return time_cost



def local_valid_test(local_model, TargetC_config, target_data_loader):
    """
    Valid the local model in the target client with validation set
    Args:
        local_model: local model
        TargetC_config: local config
        target_data_loader: local dataloader, whether validation or testing it depends
    """

    device = (
        str("cuda:" + TargetC_config["device"]) if torch.cuda.is_available() else "cpu"
    )
    local_model.to(device)
    local_model.eval()
    valid_loss = 0
    correct = 0

    for (data, labels) in target_data_loader:
        data, labels = data.to(device), labels.to(device)
        log_probs = local_model(data)

        # sum up batch loss
        valid_loss += F.nll_loss(log_probs, labels, reduction="sum").item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

    valid_loss /= len(target_data_loader.dataset)
    valid_acc = 100.00 * float(correct) / len(target_data_loader.dataset)

    return valid_loss, valid_acc


