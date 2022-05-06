import os

import torch

import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils

def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=2)


    return model


# def main(parser_data):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "val": transforms.Compose([transforms.ToTensor()])
}

VOC_root = "./"
assert os.path.exists(os.path.join(VOC_root, "VOCdevkit")), "not found VOCdevkit in path:'{}'".format(VOC_root)
# load test data set
val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=val_data_set.collate_fn)
model = create_model(num_classes=2)

train_weights = "./save_weights/resNetFpn-model-10.pth"
model.load_state_dict(torch.load(train_weights)["model"])
model.to(device)
val_mAP = []

utils.evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)




if __name__ == "__main__":
    version = torch.version.__version__[:5]  # example: 1.6.0
    if version < "1.6.0":
        raise EnvironmentError("pytorch version must be 1.6.0 or above")

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-path', default='./', help='dataset')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=65, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
