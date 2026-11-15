import torch
from torch.autograd import Variable
import utils
from args_fusion import args
import numpy as np
import time
from model import DDcGAN


def load_model(path):

    SwinFuse_model = DDcGAN(if_train=False)
    SwinFuse_model.load_state_dict(torch.load(path), False)

    para = sum([np.prod(list(p.size())) for p in SwinFuse_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(SwinFuse_model._get_name(), para * type_size / 1000 / 1000))

    SwinFuse_model.eval()
    SwinFuse_model.cuda()

    return SwinFuse_model


def run_demo(model, ir_base_path, ir_detail_path, vis_base_path, vis_detail_path, ir_path, vis_path, output_path_root, index):
    img_ir_base, h, w, c = utils.get_test_images(ir_base_path)
    img_ir_detail, h, w, c = utils.get_test_images(ir_detail_path)
    img_vis_base, h, w, c = utils.get_test_images(vis_base_path)
    img_vis_detail, h, w, c = utils.get_test_images(vis_detail_path)
    img_ir, h, w, c = utils.get_test_images(ir_path)
    img_vis, h, w, c = utils.get_test_images(vis_path)

    if c is 0:
        if args.cuda:
            img_ir_base = img_ir_base.cuda()
            img_ir_detail = img_ir_detail.cuda()
            img_vis_base = img_vis_base.cuda()
            img_vis_detail = img_vis_detail.cuda()
            img_ir = img_ir.cuda()
            img_vis = img_vis.cuda()
        img_ir_base = Variable(img_ir_base, requires_grad=False)
        img_ir_detail = Variable(img_ir_detail, requires_grad=False)
        img_vis_base = Variable(img_vis_base, requires_grad=False)
        img_vis_detail = Variable(img_vis_detail, requires_grad=False)
        img_ir = Variable(img_ir, requires_grad=False)
        img_vis = Variable(img_vis, requires_grad=False)
        img_fusion = model(img_ir_base, img_ir_detail, img_vis_base, img_vis_detail, img_ir, img_vis)
        img_fusion = ((img_fusion / 2) + 0.5) * 255
    else:
        img_fusion_blocks = []
        for i in range(c):
            img_ir_base_temp = img_ir_base[i]
            img_ir_detail_temp = img_ir_detail[i]
            img_vis_base_temp = img_vis_base[i]
            img_vis_detail_temp = img_vis_detail[i]
            img_ir_temp = img_ir[i]
            img_vis_temp = img_vis[i]

            if args.cuda:
                img_ir_base_temp = img_ir_base_temp.cuda()
                img_ir_detail_temp = img_ir_detail_temp.cuda()
                img_vis_base_temp = img_vis_base_temp.cuda()
                img_vis_detail_temp = img_vis_detail_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
                img_vis_temp = img_vis_temp.cuda()
            img_ir_base_temp = Variable(img_ir_base_temp, requires_grad=False)
            img_ir_detail_temp = Variable(img_ir_detail_temp, requires_grad=False)
            img_vis_base_temp = Variable(img_vis_base_temp, requires_grad=False)
            img_vis_detail_temp = Variable(img_vis_detail_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)
            img_vis_temp = Variable(img_vis_temp, requires_grad=False)

            img_fusion = model(img_ir_base_temp, img_ir_detail_temp, img_vis_base_temp, img_vis_detail_temp, img_ir_temp, img_vis_temp)
            img_fusion = ((img_fusion / 2) + 0.5) * 255
            img_fusion_blocks.append(img_fusion)

        if 128 < h < 256 and 128 < w < 256:
            img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)
        if 256 < h < 384 and 256 < w < 384:
            img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
        if 256 < h < 384 and 384 < w < 512:
            img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)
        if 128 < h < 256 and 256 < w < 384:
            img_fusion_list = utils.recons_fusion_images4(img_fusion_blocks, h, w)
        if 384 < h < 512 and 512 < w < 640:
            img_fusion_list = utils.recons_fusion_images5(img_fusion_blocks, h, w)
        if 0 < h < 128 and 128 < w < 256:
            img_fusion_list = utils.recons_fusion_images6(img_fusion_blocks, h, w)
        if 0 < h < 128 and 256 < w < 384:
            img_fusion_list = utils.recons_fusion_images7(img_fusion_blocks, h, w)
        if h == 128 and 256 < w < 384:
            img_fusion_list = utils.recons_fusion_images8(img_fusion_blocks, h, w)

    for img_fusion in img_fusion_list:
        file_name = str(index) + '.bmp'
        output_path = output_path_root + file_name
        utils.save_image_test(img_fusion, output_path)
        print(output_path)

def main():
    # run demo
    Ir_base_path = args.ir_base_path
    Ir_detail_path = args.ir_detail_path
    Vis_base_path = args.vis_base_path
    Vis_detail_path = args.vis_detail_path
    Output_path = args.output_path
    Ir_path = args.ir_path
    Vis_path = args.vis_path

    model_path = args.model_path_gray

    with torch.no_grad():

        model = load_model(model_path)
        begin = time.time()
        for i in range(3):
            index = i + 1
            ir_base_path = Ir_base_path + str(index) + '.bmp'
            ir_detail_path = Ir_detail_path + str(index) + '.bmp'
            vis_base_path = Vis_base_path + str(index) + '.bmp'
            vis_detail_path = Vis_detail_path + str(index) + '.bmp'
            ir_path = Ir_path + str(index) + '.bmp'
            vis_path = Vis_path + str(index) + '.bmp'

            run_demo(model, ir_base_path, ir_detail_path, vis_base_path, vis_detail_path, ir_path, vis_path, Output_path, index)
        end = time.time()
        print("consumption time of generating:%s " % (end - begin))
    print('Done......')


if __name__ == '__main__':
    main()


    def run_demo(model, ir_base_path, ir_detail_path, vis_base_path, vis_detail_path, ir_path, vis_path,
                 output_path_root, index):
        img_ir_base, h, w, c = utils.get_test_images(ir_base_path)
        img_ir_detail, h, w, c = utils.get_test_images(ir_detail_path)
        img_vis_base, h, w, c = utils.get_test_images(vis_base_path)
        img_vis_detail, h, w, c = utils.get_test_images(vis_detail_path)
        img_ir, h, w, c = utils.get_test_images(ir_path)
        img_vis, h, w, c = utils.get_test_images(vis_path)

        # ✅ 修复 1: 使用 == 而不是 is
        if c == 0:
            if args.cuda:
                img_ir_base = img_ir_base.cuda()
                img_ir_detail = img_ir_detail.cuda()
                img_vis_base = img_vis_base.cuda()
                img_vis_detail = img_vis_detail.cuda()
                img_ir = img_ir.cuda()
                img_vis = img_vis.cuda()
            img_ir_base = Variable(img_ir_base, requires_grad=False)
            img_ir_detail = Variable(img_ir_detail, requires_grad=False)
            img_vis_base = Variable(img_vis_base, requires_grad=False)
            img_vis_detail = Variable(img_vis_detail, requires_grad=False)
            img_ir = Variable(img_ir, requires_grad=False)
            img_vis = Variable(img_vis, requires_grad=False)

            img_fusion = model(img_ir_base, img_ir_detail, img_vis_base, img_vis_detail, img_ir, img_vis)
            img_fusion = ((img_fusion / 2) + 0.5) * 255

            # ✅ 关键修复：c==0 时，创建单元素列表
            img_fusion_list = [img_fusion]  # ← 这样 for 循环就能用了！

        else:
            img_fusion_blocks = []
            for i in range(c):
                # ...（中间代码保持不变）...
                img_fusion = model(...)
                img_fusion = ((img_fusion / 2) + 0.5) * 255
                img_fusion_blocks.append(img_fusion)

            # 根据尺寸重建图像列表
            img_fusion_list = None
            if 128 < h < 256 and 128 < w < 256:
                img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)
            elif 256 < h < 384 and 256 < w < 384:
                img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
            elif 256 < h < 384 and 384 < w < 512:
                img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)
            elif 128 < h < 256 and 256 < w < 384:
                img_fusion_list = utils.recons_fusion_images4(img_fusion_blocks, h, w)
            elif 384 < h < 512 and 512 < w < 640:
                img_fusion_list = utils.recons_fusion_images5(img_fusion_blocks, h, w)
            elif 0 < h < 128 and 128 < w < 256:
                img_fusion_list = utils.recons_fusion_images6(img_fusion_blocks, h, w)
            elif 0 < h < 128 and 256 < w < 384:
                img_fusion_list = utils.recons_fusion_images7(img_fusion_blocks, h, w)
            elif h == 128 and 256 < w < 384:
                img_fusion_list = utils.recons_fusion_images8(img_fusion_blocks, h, w)
            else:
                # 默认情况：直接使用 blocks（避免 None）
                img_fusion_list = img_fusion_blocks

        # ✅ 现在 img_fusion_list 一定有值！
        for img_fusion in img_fusion_list:
            file_name = str(index) + '.bmp'
            output_path = output_path_root + file_name
            utils.save_image_test(img_fusion, output_path)
            print(output_path)