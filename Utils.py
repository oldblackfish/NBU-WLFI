import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import h5py


class MyTrainSetLoader_Kfold(Dataset):
    def __init__(self, dataset_dir, test_scene_id):
        super(MyTrainSetLoader_Kfold, self).__init__()
        self.dataset_dir = dataset_dir
        scene_list = ['I01C00', 'I02C00', 'I03C00', 'I04C00', 'I05C00',
                      'I06C00', 'I07C00', 'I08C00', 'I09C00', 'I10C00']
        scene_list.pop(test_scene_id[0])
        scene_list.pop(test_scene_id[1] - 1)
        all_patch_path = []
        for scene in scene_list:
            distorted_scene_list = os.listdir(dataset_dir + '/' + scene)
            for distorted_scene in distorted_scene_list:
                distorted_path_list = os.listdir(dataset_dir + '/' + scene + '/' + distorted_scene)
                for distorted_path in distorted_path_list:
                    path = scene + '/' + distorted_scene + '/' + distorted_path
                    all_patch_path.append(path)

        self.all_patch_path = all_patch_path
        self.item_num = len(self.all_patch_path)

    def __getitem__(self, index):
        all_patch_path = self.all_patch_path
        dataset_dir = self.dataset_dir
        file_name = dataset_dir + '/' + all_patch_path[index]
        with h5py.File(file_name, 'r') as hf:
            EPPV_h = np.array(hf.get('data_h'))
            EPPV_h = EPPV_h / 255
            EPPV_h = np.transpose(EPPV_h, [1, 2, 0])

            EPPV_v = np.array(hf.get('data_v'))
            EPPV_v = EPPV_v / 255
            EPPV_v = np.transpose(EPPV_v, [1, 2, 0])

            score_label = np.array(hf.get('score_label'))

        return ToTensor()(EPPV_h.copy()), ToTensor()(EPPV_v.copy()), ToTensor()(score_label.copy())

    def __len__(self):
        return self.item_num
