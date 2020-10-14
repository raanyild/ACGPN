import time
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import os.path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from PIL import Image
import json
from PIL import ImageDraw

SIZE = 320
NC = 14



def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def get_transform(opt, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [256, 192]
        transform_list.append(transforms.Scale(osize, method))
    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base, method)))
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)





def create_edge(C_path, E_path):
    img = cv2.imread(C_path, 0)
    _, t_img = cv2.threshold(img,245,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(E_path, t_img)
    return




#for loading test data
def load_data(opt, image_name):
        #parameters
        fine_height = 256
        fine_width = 192
        radius = 5

        # get names from the pairs file
        A_path = os.path.join("/home/indranil/vtryon/ACGPN/test_data/label/" , image_name.replace(".jpg", ".png"))
        B_path = os.path.join("/home/indranil/vtryon/ACGPN/test_data/image/", image_name)
        C_path = os.path.join("/home/indranil/vtryon/ACGPN/test_data/color/", image_name)
        E_path = os.path.join("/home/indranil/vtryon/ACGPN/test_data/edge/", image_name.replace(".jpg", ".png"))
        pose_name = A_path.replace('.png', '_keypoints.json').replace(
            'label', 'pose')

        
        #load lebel
        A = Image.open(A_path).convert('L')
        if opt.label_nc == 0:
            print("hi1")
            transform_A = get_transform(opt, params)
            A_tensor = transform_A(Aconvert('RBG'))
        else:
            transform_A = get_transform(
                opt, method=Image.NEAREST, normalize=False)
            print("hi2")
            A_tensor = transform_A(A) * 255.0
            print(A_tensor)

        B_tensor = 0
        # input B (real images)
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(opt)
        B_tensor = transform_B(B)

        ### input_C (color)
        # print(self.C_paths)
        # C_path = self.C_paths[test]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        # Edge
        # E_path = self.E_paths[test]
        #create edge image and save
        
        # print(E_path)
        create_edge(C_path, E_path)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        # Pose
        with open(os.path.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, fine_height, fine_width)
        r = radius
        im_pose = Image.new('L', (fine_width, fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (fine_width, fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx +
                                r, pointy+r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor = pose_map

        A_tensor = torch.unsqueeze(A_tensor, 0)
        B_tensor = torch.unsqueeze(B_tensor, 0)
        C_tensor = torch.unsqueeze(C_tensor, 0)
        E_tensor = torch.unsqueeze(E_tensor, 0)
        P_tensor = torch.unsqueeze(P_tensor, 0)
        print(A_tensor.size())
        print(B_tensor.size())
        print(C_tensor.size())
        print(E_tensor.size())
        print(P_tensor.size())
        input_dict = {'label': A_tensor, 'image': B_tensor,
                      'edge': E_tensor, 'color': C_tensor, 
                      'pose': P_tensor}

        return input_dict




def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], NC))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img*(1-mask)
    M_c = (1-mask.cuda())*M_f
    M_c = M_c+torch.zeros(img.shape).cuda()  # broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    masked_label = label*(1-mask)
    masked_edge = mask*edge
    masked_color_strokes = mask*(1-color_mask)*color
    masked_noise = mask*noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise



def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((old_label.cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label





def main():
    os.makedirs('sample', exist_ok=True)
    opt = TestOptions().parse()

    model = create_model(opt)
    data = load_data(opt, "1.jpg")
    # add gaussian noise channel
    # wash the label
    t_mask = torch.FloatTensor(
        (data['label'].cpu().numpy() == 7).astype(np.float))
    #
    # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
    mask_clothes = torch.FloatTensor(
        (data['label'].cpu().numpy() == 4).astype(np.int))
    mask_fore = torch.FloatTensor(
        (data['label'].cpu().numpy() > 0).astype(np.int))
    img_fore = data['image'] * mask_fore
    img_fore_wc = img_fore * mask_fore
    all_clothes_label = changearm(data['label'])


    #debug
    print("debug>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(data['label'].size())
    print(data['edge'].size())
    print(data['color'].size())
    print(data['image'].size())
    print(data['pose'].size())
    print(img_fore.size())
    print(mask_clothes.size())
    print(all_clothes_label.size())
    print(mask_fore.size())
    print(data['label'])
    #debug
    ############## Forward Pass ######################
    fake_image, warped_cloth, refined_cloth = model(Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()), Variable(
        mask_clothes.cuda()), Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()), Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

    # make output folders
    output_dir = os.path.join(opt.results_dir, opt.phase)
    fake_image_dir = os.path.join(output_dir, 'try-on')
    os.makedirs(fake_image_dir, exist_ok=True)
    warped_cloth_dir = os.path.join(output_dir, 'warped_cloth')
    os.makedirs(warped_cloth_dir, exist_ok=True)
    refined_cloth_dir = os.path.join(output_dir, 'refined_cloth')
    os.makedirs(refined_cloth_dir, exist_ok=True)

    # save output
    util.save_tensor_as_image(fake_image[0],
                                os.path.join(fake_image_dir, "t1.png"))
    util.save_tensor_as_image(warped_cloth[0],
                                os.path.join(warped_cloth_dir, "t1.png"))
    util.save_tensor_as_image(refined_cloth[0],
                                    os.path.join(refined_cloth_dir, "t1.png"))





if __name__ == '__main__':
    main()
