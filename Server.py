import copy
import time
import torch
import os
from log import get_logger


def server(Common_config, client_config_list, idx_fold, temp_local_model_path, base_dir_client_weights, queue_network):
    """
    The aggregation operation in the server
    Args:
        Common_config: the common config
        client_config_list: the list of all clients' config
        queue_network: the queues connecting all clients to the server
    """

    # get the name of global layers from the target client (idx=0) at round 0.
    while True:
        if queue_network.client_has_data(0):
            w_glob_keys = queue_network.get_from_client(0)
            break
        time.sleep(0.5)

    client_weights_allrounds_list = []
    global_layers_filename_list = []
    for idx_round in range(Common_config["rounds"]):
        get_logger().info("Server: ready for round {} aggregation.".format(idx_round))
        if idx_round == 0:
            for idx_client in range(Common_config["num_clients"]):
                while True:
                    # Once the client finish local training, get the updated global layers
                    if queue_network.client_has_data(idx_client):
                        global_layers = queue_network.get_from_client(idx_client)
                        global_layers_filename_list.append(global_layers)
                        break
                    time.sleep(0.5)
        else:
            for idx_client in range(Common_config["num_clients"]):
                while True:
                    # Once the client finish local training, get the updated global layers
                    if queue_network.client_has_data(idx_client):
                        _ = queue_network.get_from_client(idx_client)
                        break
                    time.sleep(0.5)

        # load the dictionary from the file
        global_layers_list = []
        for filename in global_layers_filename_list:
            temp_dict = torch.load(filename, map_location=torch.device('cpu'))
            global_layers_list.append(temp_dict)

        global_layer_model_weights, client_weights_list = server_aggregation(
            Common_config, global_layers_list, w_glob_keys, client_config_list
        )
        client_weights_allrounds_list.append(client_weights_list)


        # save the updated global models in the files and sent back to clients
        global_layer_model_weights_save_path = os.path.join(temp_local_model_path, "global_layers.pt")
        torch.save(global_layer_model_weights, global_layer_model_weights_save_path)
        for idx_client in range(Common_config["num_clients"]):
            queue_network.send_to_client(global_layer_model_weights_save_path, idx_client)


    Client_weights_allsubs_save_path = os.path.join(base_dir_client_weights, "Clients_weights_Fold{}.txt".format(idx_fold+1))
    txtfile = open(Client_weights_allsubs_save_path, 'w')
    for lines in client_weights_allrounds_list:
        txtfile.write(str(lines))
        txtfile.write("\n")
    txtfile.close()



def server_aggregation(Common_config, local_state_dicts, w_glob_keys, client_config_list):
    """
    The server does the aggregation on the global layers' weights with different strategy
    Args:
        Common_config: the common config
        client_config_list: the list of the clients' config
        local_state_dicts: the list of the local model weights
        w_glob_keys: the names of the global layers in the client
        target_valid_dataloader_inServer: the target validation set dataloader

    Returns:
        returns the updated global layers
    """
    if Common_config["server_aggregation"] == "Fedavg":
        global_layer_model_weights, client_weights_list = Fedavg(Common_config, local_state_dicts, w_glob_keys, client_config_list)

    if Common_config["server_aggregation"] == "EqualWeights":
        global_layer_model_weights, client_weights_list = EqualWeights(Common_config, local_state_dicts, w_glob_keys)

    return global_layer_model_weights, client_weights_list


def Fedavg(Common_config, local_state_dicts, w_glob_keys, client_config_list):
    global_layer_model_weights = None
    Total_samples = 0
    client_weights = []
    for idx_client in range(Common_config["num_clients"]):
        local_sate_dict = local_state_dicts[idx_client]
        sample_num = client_config_list[idx_client]["num_samples"]
        Total_samples += sample_num
        if global_layer_model_weights is None:
            global_layer_model_weights = {}
            for k in w_glob_keys:
                global_layer_model_weights[k] = copy.deepcopy(local_sate_dict[k].cpu()) * sample_num
        else:
            for k in w_glob_keys:
                global_layer_model_weights[k] += local_sate_dict[k].cpu() * sample_num
    for k in w_glob_keys:
        global_layer_model_weights[k] = (
                global_layer_model_weights[k] / Total_samples
        )
    client_weights_list = [weights / Total_samples for weights in client_weights]

    return global_layer_model_weights, client_weights_list

def EqualWeights(Common_config, local_state_dicts, w_glob_keys):
    global_layer_model_weights = None
    client_weights = []
    for idx_client in range(Common_config["num_clients"]):
        local_sate_dict = local_state_dicts[idx_client]
        if global_layer_model_weights is None:
            global_layer_model_weights = {}
            for k in w_glob_keys:
                global_layer_model_weights[k] = copy.deepcopy(local_sate_dict[k].cpu())
        else:
            for k in w_glob_keys:
                global_layer_model_weights[k] += local_sate_dict[k].cpu()
    for k in w_glob_keys:
        global_layer_model_weights[k] = (
                global_layer_model_weights[k] / Common_config["num_clients"]
        )
    return global_layer_model_weights, client_weights


