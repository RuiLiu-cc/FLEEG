import logging
import os
import random
import numpy as np
import pandas as pd
import torch

from ArgumentsSetup import assign_arguments
from Clients import client
from log import (add_file_handler, get_logger, set_formatter, set_level, stop_logger)
from process_initialization import forward
from ProcessTaskQueue import CommunicationNework
from SaveSharedArray import Load_data_into_shared_array_allClient
from Server import server
from torch_process_pool import TorchProcessPool
from utils import ResultsSaveDirection

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

if __name__ == "__main__":

    # Configuration Initialization
    (
        Common_config,
        Client_config_list
    ) = assign_arguments()

    # set the results saving direction
    (
        Client_base_dir_list,
        base_dir_logger,
        base_dir_temp_local_models,
        base_dir_client_weights
    ) = ResultsSaveDirection(Common_config, Client_config_list)

    # Initialize the results saving array
    results_allsubs_columns = [
        "subs",
        "test_acc_validloss",
        "test_loss_validloss",
        "test_acc_validacc",
        "test_loss_validacc",
    ]
    results_allsubs_list = []
    results_allsubs_save_path_list = []
    for idx_client in range(Common_config["num_clients"]):
        results_allsubs_list.append([])
        results_allsubs_save_path = os.path.join(Client_base_dir_list[idx_client], "Allsubs_results.csv")
        results_allsubs_save_path_list.append(results_allsubs_save_path)

    # set the logger saving direction
    add_file_handler(base_dir_logger + "/logger.txt")
    set_level(logging.INFO)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    set_formatter(formatter)

    # load the data into shared array
    Data_array_list = Load_data_into_shared_array_allClient(
        Common_config,
        Client_config_list
    )

    # k-fold cross-validation training strategy for each client
    fold_partition_list = []
    for idx_client in range(Common_config["num_clients"]):
        client_config = Client_config_list[idx_client]
        temp = list(range(client_config["num_folds"]))
        if client_config["num_folds"] == Common_config["num_largest_fold"]:
            random.shuffle(temp)
            test_set = [[i] for i in temp]
            test_set = np.asarray(test_set)
        else:
            n_repeats = Common_config["num_largest_fold"] // client_config["num_folds"] + 1
            test_set_repeat = []
            for _ in range(n_repeats):
                random.shuffle(temp)
                test_set_temp = [[i] for i in temp]
                test_set_repeat = test_set_repeat + test_set_temp
            test_set = test_set_repeat[0:Common_config["num_largest_fold"]]
            test_set = np.asarray(test_set)
        fold_partition_list.append(test_set)


    # Fold itertation
    for idx_fold in range(Common_config["num_largest_fold"]):

        if idx_fold >= Common_config["num_folds"][0] and idx_fold < Common_config["num_folds"][1]:
            get_logger().info("============Train with fold %s ============", idx_fold + 1)

            queue_network = CommunicationNework(Common_config)
            pool = TorchProcessPool(
                initargs=[[], {}, {"queue_network": queue_network}],
            )

            # spaw the process for the server
            pool.submit(
                forward,
                server,
                Common_config,
                Client_config_list,
                idx_fold,
                base_dir_temp_local_models,
                base_dir_client_weights
            )

            # spaw the process for the clients
            client_future_list = []
            for idx_client in range(Common_config["num_clients"]):

                # get the config & results saving folder direction for the current client
                client_config = Client_config_list[idx_client]
                Client_base_dir = Client_base_dir_list[idx_client]

                # get the subject index of training and testing set for the current fold
                client_fold_partition_idx = fold_partition_list[idx_client]
                test_index = client_fold_partition_idx[idx_fold]

                # spaw the process for the client
                client_future = pool.submit(
                    forward,
                    client,
                    idx_client,
                    idx_fold,
                    test_index,
                    Client_base_dir,
                    base_dir_temp_local_models,
                    Common_config,
                    client_config,
                    Data_array_list[idx_client],
                )
                client_future_list.append(client_future)

            pool.wait_results()
            for idx_client in range(Common_config["num_clients"]):
                current_client_future = client_future_list[idx_client]
                current_client_results_allsubs = results_allsubs_list[idx_client]
                (
                    current_client_best_test_acc_list,
                    current_client_best_test_loss_list,
                    current_client_optimal_test_acc_list,
                    current_client_optimal_test_loss_list
                ) = current_client_future.result()

                # Record the best results for the current fold
                client_fold_partition_idx = fold_partition_list[idx_client]
                test_index = client_fold_partition_idx[idx_fold]
                for idx in range(len(test_index)):
                    current_client_results_allsubs.append(
                        np.array(
                            [
                                test_index[idx] + 1,
                                current_client_best_test_acc_list[idx],
                                current_client_best_test_loss_list[idx],
                                current_client_optimal_test_acc_list[idx],
                                current_client_optimal_test_loss_list[idx],
                            ]
                        )
                    )
            pool.shutdown()


    # Add the fold results into the final tabel
    for idx_client in range(Common_config["num_clients"]):
        results_allsubs = results_allsubs_list[idx_client]
        results_allsubs_save_path = results_allsubs_save_path_list[idx_client]

        final_results_allsubs = np.array(results_allsubs)
        final_results_allsubs = pd.DataFrame(
            final_results_allsubs, columns=results_allsubs_columns
        )

        # calculate the averaged accuracy
        col_mean = final_results_allsubs[
            ["test_acc_validloss",
            "test_loss_validloss",
            "test_acc_validacc",
            "test_loss_validacc",]
        ].mean()
        final_results_allsubs = final_results_allsubs.append(col_mean, ignore_index=True)
        final_results_allsubs.to_csv(results_allsubs_save_path, index=False)

    get_logger().warning("end of experiments")
    stop_logger()
