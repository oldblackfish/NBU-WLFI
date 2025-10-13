import torch
from Utils import *
from Model import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SRCC


def val(valset_dir, test_scene_id, load_model_path, type='val'):
    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True
    model = torch.load(load_model_path, map_location={'cuda:0': device})
    net.load_state_dict(model['state_dict'])
    net.eval()

    label_list = []
    data_list = []
    scene_list = ['I01C00', 'I02C00', 'I03C00', 'I04C00', 'I05C00',
                  'I06C00', 'I07C00', 'I08C00', 'I09C00', 'I10C00']
    for test_scene in test_scene_id:
        image_path = valset_dir + '/' + scene_list[test_scene]
        image_list = os.listdir(image_path)
        for test_image in image_list:
            patch_path = image_path + '/' + test_image
            patch_list = os.listdir(patch_path)
            output_list = 0
            for val_patch in patch_list:
                each_patch_path = patch_path + '/' + val_patch
                with h5py.File(each_patch_path, 'r') as hf:
                    label = np.array(hf.get('score_label'))
                    EPPV_h = np.array(hf.get('data_h'))
                    EPPV_h = EPPV_h / 255
                    EPPV_h = np.expand_dims(EPPV_h, axis=0)
                    EPPV_h = np.expand_dims(EPPV_h, axis=0)
                    EPPV_h = torch.from_numpy(EPPV_h.copy())
                    EPPV_h = Variable(EPPV_h).to(device)

                    EPPV_v = np.array(hf.get('data_v'))
                    EPPV_v = EPPV_v / 255
                    EPPV_v = np.expand_dims(EPPV_v, axis=0)
                    EPPV_v = np.expand_dims(EPPV_v, axis=0)
                    EPPV_v = torch.from_numpy(EPPV_v.copy())
                    EPPV_v = Variable(EPPV_v).to(device)
                with torch.no_grad():
                    out_score = net(EPPV_h, EPPV_v)
                output_list += out_score.cpu().numpy().item()
            label_list.append(label.item())
            data_list.append(output_list / len(patch_list))

    val_SRCC = SRCC(data_list, label_list).correlation
    print(type + ' SRCC :----    %f' % val_SRCC)
