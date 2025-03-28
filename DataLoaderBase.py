import random

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ArrayDataLoader(Dataset):
    def __init__(self, dataset):
        super().__init__()
        """
        Load the data from a list of shared array into the "Dataset" format
        Can be used as the test dataloader and other clients's dataloader
        Parameters:
            dataset: a list of shared arrays [shared_data, shared_label]
            shared_data: (#trials, 1, #channels, #timesteps)
            shared_label: (#trials, )
        """
        self.x_data = dataset[0]
        self.y_data = dataset[1]
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# ==============================================
class SplitTrain(Dataset):
    def __init__(self, dataset, test_index, Common_config):
        super().__init__()
        """
        load the data from several subjects and select part of them as training set and stored as the "Dataset" format
        Param:
            dataset: list of array, [TarC_Sub1_data_shared_array, TarC_Sub2_shared_array, ..., TarC_SubN_shared_array]
            test_index: list, index of the subject selected as the testing subject
            Common_config: common config
            Common_config['subject_wise']: boolean, subject-wise (True) or trial_wise (False) partition into train and valid set.
            Common_config.['split_ratio']: float, ratio of valid set.
        """
        # Leave the selected subject out
        subs_idx_list = list(range(len(dataset)))
        for idx in test_index:
            subs_idx_list.remove(idx)

        if Common_config["subject_wise"]:
            random.seed(42)
            train_subs = random.sample(
                subs_idx_list,
                int((1 - Common_config["split_ratio"]) * len(subs_idx_list)),
            )
            self.x_data = dataset[train_subs[0]][0]
            self.y_data = dataset[train_subs[0]][1]

            for sub in train_subs[1:]:
                self.x_data = torch.cat((self.x_data, dataset[sub][0]), dim=0)
                self.y_data = torch.cat((self.y_data, dataset[sub][1]), dim=0)

        else:
            self.x_data = dataset[subs_idx_list[0]][0]
            self.y_data = dataset[subs_idx_list[0]][1]

            for sub in subs_idx_list[1:]:
                self.x_data = torch.cat((self.x_data, dataset[sub][0]), dim=0)
                self.y_data = torch.cat((self.y_data, dataset[sub][1]), dim=0)

            self.x_data, _, self.y_data, _ = train_test_split(
                self.x_data,
                self.y_data,
                test_size=Common_config["split_ratio"],
                random_state=42,
            )

        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class SplitValid(Dataset):
    def __init__(self, dataset, test_index, Common_config):
        super().__init__()
        """
        load the data from several subjects and select part of them as validation set and stored as the "Dataset" format
        Param:
            dataset: list of array, [TarC_Sub1_data_shared_array, TarC_Sub2_shared_array, ..., TarC_SubN_shared_array]
            test_index: list, index of the subject selected as the testing subject
            Common_config: common config
            Common_config['subject_wise']: boolean, subject-wise (True) or trial_wise (False) partition into train and valid set.
            Common_config['split_ratio']: float, ratio of valid set.
        """
        # Leave the selected subject out
        subs_idx_list = list(range(len(dataset)))
        for idx in test_index:
            subs_idx_list.remove(idx)

        if Common_config["subject_wise"]:
            random.seed(42)
            train_subs = random.sample(
                subs_idx_list,
                int((1 - Common_config["split_ratio"]) * len(subs_idx_list)),
            )
            valid_subs = list(set(subs_idx_list) - set(train_subs))

            self.x_data = dataset[valid_subs[0]][0]
            self.y_data = dataset[valid_subs[0]][1]

            for sub in valid_subs[1:]:
                self.x_data = torch.cat((self.x_data, dataset[sub][0]), dim=0)
                self.y_data = torch.cat((self.y_data, dataset[sub][1]), dim=0)

        else:
            self.x_data = dataset[subs_idx_list[0]][0]
            self.y_data = dataset[subs_idx_list[0]][1]

            for sub in subs_idx_list[1:]:
                self.x_data = torch.cat((self.x_data, dataset[sub][0]), dim=0)
                self.y_data = torch.cat((self.y_data, dataset[sub][1]), dim=0)

            _, self.x_data, _, self.y_data = train_test_split(
                self.x_data,
                self.y_data,
                test_size=Common_config["split_ratio"],
                random_state=42,
            )

        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
