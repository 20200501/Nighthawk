import os

import torch

import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils
from predict import predict


def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # data_transform = {
    #     "train": transforms.Compose([transforms.ToTensor(),
    #                                  transforms.RandomHorizontalFlip(0.5)]),
    #     "val": transforms.Compose([transforms.ToTensor()])
    # }

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "UIDISPLAYISSUE")) is False:
        raise FileNotFoundError("UIDISPLAYISSUE dose not in path:'{}'.".format(VOC_root))


    # load train data set
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)

    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=2)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=7,
                                                   gamma=0.75)

    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_mAP = []

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, train_loss=train_loss, train_lr=learning_rate,
                              print_freq=50, warmup=True)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

        predict(epoch)

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        from plot_curve import plot_map
        plot_map(val_mAP)

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    # predictions = model(x)
    # print(predictions)


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
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=5, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
