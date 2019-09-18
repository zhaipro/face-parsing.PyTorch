#!/usr/bin/python
# -*- encoding: utf-8 -*-
import cv2
import torch
import torchvision.transforms as transforms

from model import BiSeNet


def evaluate(net, image):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        im = cv2.resize(image, (512, 512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = to_tensor(im)
        im = torch.unsqueeze(im, 0)
        im = im.cuda()
        out = net(im)[0]
        parsing = out.squeeze(0).cpu().numpy()
        print(parsing.shape)
        return parsing


if __name__ == '__main__':
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = 'res/cp/79999_iter.pth'
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    image = cv2.imread('data/116_ori.jpg')
    parsing = evaluate(net, image)
    import numpy as np
    np.save('parsing.npy', parsing)
