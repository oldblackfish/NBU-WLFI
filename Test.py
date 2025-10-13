import torch
from Utils import *
from Model import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC


def test_model():
    ### NBU
    load_all_model_path = './PreTrainedModels/NBU-WLFI/'
    valset_dir = './Datasets/NBU_WLFI_5x32x64/'
    dataset_name = 'NBU_WLF1.0'
    scene_list = ['I01C00', 'I02C00', 'I03C00', 'I04C00', 'I05C00',
                  'I06C00', 'I07C00', 'I08C00', 'I09C00', 'I10C00']
    test_scene_num = 2
    distorted_num = 20
    scene_num = 10

    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True

    all_model = os.listdir(load_all_model_path)
    label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    val_SRCC_all = []
    test_scene_id_list = []
    for a in range(0, scene_num, 2):
        test_scene_id_list.append([a, a + 1])

    for id, model_name in enumerate(test_scene_id_list):
        load_model_path = load_all_model_path + '/' + \
                          str(model_name[0]) + '_' + str(model_name[1]) + '/EPPVS-BWLFQ_epoch50.pth.tar'
        model = torch.load(load_model_path, map_location={'cuda:0': device})
        net.load_state_dict(model['state_dict'])
        net.eval()
        index = 0
        test_scene_id = [int(model_name[0]), int(model_name[1])]
        for test_scene in test_scene_id:
            image_path = valset_dir + '/' + scene_list[test_scene]
            image_list = os.listdir(image_path)
            for test_image in image_list:
                patch_path = valset_dir + '/' + scene_list[test_scene] + '/' + test_image
                patch_list = os.listdir(patch_path)
                output_list = []
                for val_patch in patch_list:
                    if int(val_patch[:-3]) < int(len(patch_list) * 0.7):
                        continue
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
                    output_list.append(out_score.cpu().numpy().item())
                label_list[index, id] = label.item()
                data_list[index, id] = np.mean(output_list).item()
                index += 1

        val_SRCC = SROCC(data_list[:, id], label_list[:, id]).correlation
        val_SRCC_all.append(val_SRCC)
        print(test_scene_id)
        print('SROCC :----    %f' % val_SRCC)
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))

    # save in h5 file and test in matlab
    f = h5py.File('./Results/EPPVS-BWLFQ_result_' + dataset_name + '.h5', 'w')
    f.create_dataset('predict_data', data=data_list)
    f.create_dataset('score_label', data=label_list)
    f.close()


if __name__ == '__main__':
    test_model()
