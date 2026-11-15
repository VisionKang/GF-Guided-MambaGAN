import torch.nn as nn
from os import listdir
import numpy as np
import torch
from args_fusion import args
import imageio

def crop_to_patch(img, stride):
    h = img.shape[0]
    w = img.shape[1]
    print(h)
    print(w)
    all_img = []
    for i in range(0, h - 128, stride):
        for j in range(0, w - 128, stride):
            img_patch = img[i:i + 128, j:j + 128].reshape(1, 128, 128)
            all_img.append(img_patch)
            if i + 128 >= h:
                img_patch = img[h - 128:, j:j + 128].reshape(1, 128, 128)
                all_img.append(img_patch)
        img_patch = img[i:i + 128, w - 128:].reshape(1, 128, 128)
        all_img.append(img_patch)
    all_img = np.stack(all_img, axis=0).reshape(len(all_img), 1, 128, 128)  # (b, 1, 224, 224)
    return all_img

def list_images(dimgectory):
    dimg = listdir(dimgectory)
    dimg.sort()
    return dimg

def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]

    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_image(path):
    image = imageio.imread(path, pilmode='L')
    return image

def get_image_test(path):
    image = imageio.imread(path, pilmode='L')
    image = (image-127.5)/127.5

    return image

def get_train_images_auto(paths):
    images = []
    crop_images = []

    for path in paths:
        image = get_image(path)
        images.append(image)

    for image in images:
        crop_image = crop_to_patch(image, 20)
        crop_images.append(crop_image)

    crop_images = np.concatenate(crop_images, axis=0)
    crop_images = torch.from_numpy(crop_images).float()
    crop_images = (crop_images-127.5)/127.5
    print(len(crop_images))
    return crop_images


def get_test_images(paths):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image_test(path)
        base_size = 128
        h = image.shape[0]
        w = image.shape[1]
        if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)
        if 2 * base_size < h < 3 * base_size and 2 * base_size < w < 3 * base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)

    return images, h, w, c


def get_img_parts1(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 256-w, 0, 256-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 128]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[128:256, 0: 128]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[128:256, 128: 256]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    return images


def get_img_parts2(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 384-w, 0, 384-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:128, 256: 384]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[128:256, 0: 128]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[128:256, 128: 256]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[128:256, 256: 384]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[256:384, 0: 128]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[256:384, 128: 256]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[256:384, 256: 384]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images


def get_img_parts3(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 512-w, 0, 384-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:128, 256: 384]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:128, 384: 512]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[128:256, 0: 128]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[128:256, 128: 256]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[128:256, 256: 384]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[128:256, 384: 512]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[256:384, 0: 128]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[256:384, 128: 256]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[256:384, 256: 384]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[256:384, 384: 512]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    return images


def get_img_parts4(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 384-w, 0, 256-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:128, 256: 384]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[128:256, 0: 128]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[128:256, 128: 256]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[128:256, 256: 384]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    return images


def get_img_parts5(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 640-w, 0, 512-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:128, 256: 384]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:128, 384: 512]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[0:128, 512: 640]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[128:256, 0: 128]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[128:256, 128: 256]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[128:256, 256: 384]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[128:256, 384: 512]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[128:256, 512: 640]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[256:384, 0: 128]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[256:384, 128: 256]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    img13 = image[256:384, 256: 384]
    img13 = torch.reshape(img13, [1, 1, img13.shape[0], img13.shape[1]])
    img14 = image[256:384, 384: 512]
    img14 = torch.reshape(img14, [1, 1, img14.shape[0], img14.shape[1]])
    img15 = image[256:384, 512: 640]
    img15 = torch.reshape(img15, [1, 1, img15.shape[0], img15.shape[1]])
    img16 = image[384:512, 0: 128]
    img16 = torch.reshape(img16, [1, 1, img16.shape[0], img16.shape[1]])
    img17 = image[384:512, 128: 256]
    img17 = torch.reshape(img17, [1, 1, img17.shape[0], img17.shape[1]])
    img18 = image[384:512, 256: 384]
    img18 = torch.reshape(img18, [1, 1, img18.shape[0], img18.shape[1]])
    img19 = image[384:512, 384: 512]
    img19 = torch.reshape(img19, [1, 1, img19.shape[0], img19.shape[1]])
    img20 = image[384:512, 512: 640]
    img20 = torch.reshape(img20, [1, 1, img20.shape[0], img20.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    images.append(img13.float())
    images.append(img14.float())
    images.append(img15.float())
    images.append(img16.float())
    images.append(img17.float())
    images.append(img18.float())
    images.append(img19.float())
    images.append(img20.float())

    return images


def get_img_parts6(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 256-w, 0, 128-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    return images


def get_img_parts7(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 384-w, 0, 128-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:128, 256: 384]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def get_img_parts8(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 384-w, 0, 0), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:128, 0: 128]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:128, 128: 256]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:128, 256: 384]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def recons_fusion_images1(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:128, 0: 128] += img1
        img_f[:, 0:128, 128: w] += img2[:, 0:128, 0:w-128]
        img_f[:, 128:h, 0: 128] += img3[:, 0:h-128, 0:128]
        img_f[:, 128:h, 128: w] += img4[:, 0:h-128, 0:w-128]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images2(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:128, 0: 128] += img1
        img_f[:, 0:128, 128: 256] += img2
        img_f[:, 0:128, 256: w] += img3[:, 0:128, 0:w-256]
        img_f[:, 128:256, 0: 128] += img4
        img_f[:, 128:256, 128: 256] += img5
        img_f[:, 128:256, 256: w] += img6[:, 0:128, 0:w-256]
        img_f[:, 256:h, 0: 128] += img7[:, 0:h-256, 0:128]
        img_f[:, 256:h, 128: 256] += img8[:, 0:h-256, 0:128]
        img_f[:, 256:h, 256: w] += img9[:, 0:h-256, 0:w - 256]
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images3(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:128, 0: 128] += img1
        img_f[:, 0:128, 128: 256] += img2
        img_f[:, 0:128, 256: 384] += img3
        img_f[:, 0:128, 384: w] += img4[:, 0:128, 0:w-384]
        img_f[:, 128:256,0: 128] += img5
        img_f[:, 128:256, 128: 256] += img6
        img_f[:, 128:256, 256: 384] += img7
        img_f[:, 128:256, 384: w] += img8[:, 0:128, 0:w-384]
        img_f[:, 256:h, 0: 128] += img9[:, 0:h-256, 0:128]
        img_f[:, 256:h, 128: 256] += img10[:, 0:h - 256, 0:128]
        img_f[:, 256:h, 256: 384] += img11[:, 0:h - 256, 0:128]
        img_f[:, 256:h, 384: w] += img12[:, 0:h - 256, 0:w-384]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images4(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]

        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:128, 0: 128] += img1
        img_f[:, 0:128, 128: 256] += img2
        img_f[:, 0:128, 256: w] += img3[:, 0:128, 0:w-256]
        img_f[:, 128:h, 0: 128] += img4[:, 0:h-128, 0:128]
        img_f[:, 128:h, 128: 256] += img5[:, 0:h - 128, 0:128]
        img_f[:, 128:h, 256: w] += img6[:, 0:h - 128, 0:w-256]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images5(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img13 = img_lists[12][i]
        img14 = img_lists[13][i]
        img15 = img_lists[14][i]
        img16 = img_lists[15][i]
        img17 = img_lists[16][i]
        img18 = img_lists[17][i]
        img19 = img_lists[18][i]
        img20 = img_lists[19][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:128, 0: 128] += img1
        img_f[:, 0:128, 128: 256] += img2
        img_f[:, 0:128, 256: 384] += img3
        img_f[:, 0:128, 384: 512] += img4
        img_f[:, 0:128, 512: w] += img5[:, 0:128, 0:w-512]
        img_f[:, 128:256, 0: 128] += img6
        img_f[:, 128:256, 128: 256] += img7
        img_f[:, 128:256, 256: 384] += img8
        img_f[:, 128:256, 384: 512] += img9
        img_f[:, 128:256, 512: w] += img10[:, 0:128, 0:w-512]
        img_f[:, 256:384, 0: 128] += img11
        img_f[:, 256:384, 128: 256] += img12
        img_f[:, 256:384, 256: 384] += img13
        img_f[:, 256:384, 384: 512] += img14
        img_f[:, 256:384, 512: w] += img15[:, 0:128, 0:w - 512]
        img_f[:, 384:h, 0: 128] += img16[:, 0:h-384, 0:128]
        img_f[:, 384:h, 128: 256] += img17[:, 0:h-384, 0:128]
        img_f[:, 384:h, 256: 384] += img18[:, 0:h-384, 0:128]
        img_f[:, 384:h, 384: 512] += img19[:, 0:h-384, 0:128]
        img_f[:, 384:h, 512: w] += img20[:, 0:h-384, 0:w - 512]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images6(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 128] += img1[:, 0:h, 0:128]
        img_f[:, 0:h, 128: w] += img2[:, 0:h, 0:w-128]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images7(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 128] += img1[:, 0:h, 0:128]
        img_f[:, 0:h, 128: 256] += img2[:, 0:h, 0:128]
        img_f[:, 0:h, 256: w] += img3[:, 0:h, 0:w - 256]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images8(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 128] += img1[:, 0:h, 0:128]
        img_f[:, 0:h, 128: 256] += img2[:, 0:h, 0:128]
        img_f[:, 0:h, 256: w] += img3[:, 0:h, 0:w - 256]

        img_f_list.append(img_f)
    return img_f_list

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    print(img_fusion.shape)
    img_fusion = img_fusion.reshape([1, img_fusion.shape[0], img_fusion.shape[1]])
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imageio.imwrite(output_path, img_fusion)
