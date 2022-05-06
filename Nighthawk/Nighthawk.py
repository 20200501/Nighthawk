import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function, Variable
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from network import Net  
from network import GradCam  
import torch.nn as nn
import getdata
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from threading import Timer
import subprocess

def parse_args():
    """
    parse command line input
    generate options including host name, port number
    """
    parser = argparse.ArgumentParser(description="Turn on your emulator to start testing.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("-d", action="store", dest="device_serial", required=False,
                        help="The serial number of target device (use `adb devices` to find)")
    
    parser.add_argument("-a", action="store", dest="apk_path", required=True,
                        help="The file path to target APK")
    
    parser.add_argument("-o", action="store", dest="output_dir",
                        help="directory of output")

    parser.add_argument("-t", action="store", dest="duration", 
                        help="duration of running the emulator for generating screenshot.")
    options = parser.parse_args()
    return options

ALLOWED_APK_EXTENSIONS = ['apk']
ALLOWED_IMAGE_EXTENSIONS = ['jpeg', 'jpg', 'png']

model_dir = './model/4model.pth' 

def killing_robot(proc):
    proc.terminate()

def main():
    opts = parse_args()
    import os
    if not os.path.exists(opts.apk_path):
        print("APK does not exist.")
        return
    
    proc = subprocess.Popen(['droidbot', '-a', opts.apk_path, '-o', './temp', '-is_emulator'], shell=True)
    timer = Timer(opt.duration+360, killing_robot, args=(proc,))
    config.timer = timer
    timer.start()
    timer.join()

    output_dir = os.path.join('opts.apk_path', 'states')

    model = Net()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))

    for file in output_dir:
        if file=='.DS_Store':
            continue
        (filename, extension) = os.path.splitext(file)
        
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        
        grad_cam = GradCam(model=model, target_layer_names=["40"], use_cuda=args.use_cuda)
        img_ori = cv2.imread(image_name, 1)
        height, width, channels = img_ori.shape
        img = np.float32(cv2.resize(img_ori, (448, 768))) / 255

        input = preprocess_image(image_name)
        target_index = None
        out = model(input)
        out = F.softmax(out, dim=1)
        out = out.data.cpu().numpy()
        out2 = out[0]
        print(out2)
        bug = out2[0] > out2[1]

        mask = grad_cam(input, target_index)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        real_cam = cv2.resize(cam, (width, height))

        ori_path = os.path.join(opts.output_dir,'{0}.jpg'.format(image_num))
        cam_spath = os.path.join(opts.output_dir,'{0}cam_gb.jpg'.format(image_num))

        cv2.imwrite(ori_path, img_ori)
        cv2.imwrite(cam_path, cam_gb)

        if bug:
            print('bug pair detected!',
            "original screenshot path: "+ori_path,
            "cam picture path: "+cam_spath,
            sep='\t')
        else:
            print('clean pair detected!',
            "original screenshot path: "+ori_path,
            "cam picture path: "+cam_spath,
            sep='\t')
        

