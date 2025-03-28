import ctypes
import torch.multiprocessing as mp
import os
import numpy as np
import torch

# Revised from ptrblck's
# codes here: https://github.com/ptrblck/pytorch_misc/blob/master/shared_array.py and my previous code in DataLoaderBase.py

def data_load(data_path):
    """
    Load the data file name into a list
    Params:
        data_path: str, data path contains all filtered EEG signal
    Return:
        a list,  a list of file names stored in the given data path
    """

    files = os.listdir(data_path)
    return sorted(files)


def Load_data_into_shared_array_allClient(Common_config, Client_config_list):
    """
    Load the datasets into a list with each element for one client.
    For each client, each subject's data is loaded as a shared array
    Parameters:
        Common_config: common config
        Client_config_list: the config of the all clients
    Returns:
        Data_list=
            [
            [Client1_Sub1_data_shared_array, Client1_Sub2_shared_array, ..., Client1_SubN1_shared_array],
            [Client2_Sub1_data_shared_array, Client2_Sub2_shared_array, ..., Client2_SubN2_shared_array],
            [Client3_Sub1_data_shared_array, Client3_Sub2_shared_array, ..., Client3_SubN3_shared_array],
             ...,
             [Clientk_Sub1_data_shared_array, Clientk_Sub2_shared_array, ..., Clientk_SubNk_shared_array],
             ]
        Each "xxx_data_shared_array" = [shared_data, shared_label]
    """
    Data_list = []
    for client_config in Client_config_list:
        Data_list.append(Target_client(Common_config, client_config))

    return Data_list



def Target_client(Common_config, TargetC_config):
    """
    Load each subject's data as a shared array
    Args:
        Common_config: common config
        TargetC_config: the config of the target client

    Returns:
        Target_data:
        [TarC_Sub1_data_shared_array, TarC_Sub2_shared_array, ..., TarC_SubN_shared_array]
        In TarC_Sub1_data_shared_array:
        [shared_data, shared_label]
        shared_data: (#trials, 1, #channels, #timesteps)
        shared_label: (#trials, )
    """
    Target_data = []
    client_data_path = os.path.join(
        Common_config["data_path"],
        TargetC_config["name"],
        TargetC_config["filtering_setting"],
    )
    print(client_data_path)
    files_name_list = data_load(client_data_path)

    for file in files_name_list:
        print(file)
        file_name = os.path.join(client_data_path, file)
        onesub_data = np.load(file_name)
        x_data = onesub_data["x_data"]
        y_data = onesub_data["y_data"]
        # covert the data size into (#trials, 1, #channel, #timesteps) to fit the conv2d
        x_data = np.expand_dims(x_data, axis=1)
        num_trials = x_data.shape[0]
        onesub_data.close()

        shared_array_base = mp.Array(
            ctypes.c_float, x_data.shape[0] * 1 * x_data.shape[2] * x_data.shape[3]
        )
        shared_data = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_data = shared_data.reshape(
            x_data.shape[0], 1, x_data.shape[2], x_data.shape[3]
        )
        shared_data = torch.from_numpy(shared_data)

        # label size is (#trials, )
        shared_array_label_base = mp.Array(ctypes.c_long, num_trials)
        shared_label = np.ctypeslib.as_array(shared_array_label_base.get_obj())
        shared_label = shared_label.reshape(num_trials, 1)
        shared_label = torch.from_numpy(shared_label)

        # load the data into the shared array
        for index in range(num_trials):
            shared_data[index] = torch.from_numpy(x_data[index, :, :, :]).float()
        shared_label[:, 0] = torch.from_numpy(y_data).long()
        shared_label = shared_label.reshape(num_trials)

        del x_data, y_data
        Target_data.append([shared_data, shared_label])

    print("=============================")

    return Target_data



