import os
#current_path = os.path.normpath(os.getcwd())
#current_path = current_path.split(os.sep)
#str_path = "/".join(current_path[:-2])
import sys
import shutil
from tqdm import *
#sys.path.append(str_path)
#os.chdir(str_path)
#print(str_path)
import datetime
import os
from scipy.io import savemat
from math import sqrt
import numpy as np
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets.blood_cells import *
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import cv2
from utils.segmentation_vizualizer import *
from unet.unet_model import UNet
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from torchvision import utils
ckpt_path = './ckpt'
exp_name = 'blood_cells_unet_smaller'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))
from matplotlib import pyplot as plt
args = {
    'train_batch_size': 5,
    'lr': 1e-2 / sqrt(16 / 2),
    'lr_decay': 0.95,
    'max_iter': 20000,
    'longer_size': int(256),
    "short_size": int(256),
    'crop_size': 128,  # 768,
    'stride_rate': 0.2,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_save_to_img_file': True,
    'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
    'val_img_display_size': 384,
    'val_freq': 100
}


def main():
    net = UNet(3, n_classes=3)
    #net.load_state_dict(torch.load("./MODEL.pth"))
    #print("Model loaded.")
    if len(args['snapshot']) == 0:
        # net.load_state_dict(torch.load(os.path.join(ckpt_path, 'cityscapes (coarse)-psp_net', 'xx.pth')))
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'iter': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0,
                               'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'iter': int(split_snapshot[3]),
                               'val_loss': float(split_snapshot[5]), 'acc': float(split_snapshot[7]),
                               'acc_cls': float(split_snapshot[9]),'mean_iu': float(split_snapshot[11]),
                               'fwavacc': float(split_snapshot[13])}
    net.cuda().train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(args['longer_size']),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip()
    ])
    sliding_crop = joint_transforms.SlidingCrop(args['crop_size'], args['stride_rate'], ignore_label)
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(args['val_img_display_size']),
        standard_transforms.ToTensor()
    ])

    train_set = Retinaimages('training', joint_transform=train_joint_transform, sliding_crop=sliding_crop,
                                      transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=2, shuffle=True)
    val_set = Retinaimages('validate', transform=val_input_transform, sliding_crop=sliding_crop,
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=2, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True).cuda()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, "_1" + '.txt'), 'w').write(str(args) + '\n\n')

    train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader, visualize, val_set)


def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader, visualize, val_set):
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        #validate(val_loader, net, criterion, optimizer, curr_epoch, 0 + 15000, train_args, visualize, val_set)
        #exit(-1)
        t = tqdm(train_loader)
        for i, data in enumerate(t):
            t.set_description("EPOCH: {}, best IOU: {:3f}".format(curr_epoch, train_args['best_record']['mean_iu']))
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']

            inputs, gts, _ = data
            assert len(inputs.size()) == 5 and len(gts.size()) == 4
            inputs.transpose_(0, 1)
            gts.transpose_(0, 1)

            assert inputs.size()[3:] == gts.size()[2:]
            slice_batch_pixel_size = inputs.size(1) * inputs.size(3) * inputs.size(4)

            for inputs_slice, gts_slice in zip(inputs, gts):
                inputs_slice = Variable(inputs_slice).cuda()
                gts_slice = Variable(gts_slice).cuda()

                optimizer.zero_grad()
                outputs = net(inputs_slice)
                #print(gts_slice.size()[1:])
                #print(outputs.size())

                assert outputs.size()[2:] == gts_slice.size()[1:]
                assert outputs.size()[1] == num_classes

                main_loss = criterion(outputs, gts_slice)
                #aux_loss = criterion(aux, gts_slice)

                loss = main_loss
                loss.backward()
                optimizer.step()

                train_main_loss.update(main_loss.data[0], slice_batch_pixel_size)
                #train_aux_loss.update(aux_loss.data[0], slice_batch_pixel_size)

            curr_iter += 1
            writer.add_scalar('train_main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

            # if (i + 1) % train_args['print_freq'] == 0:
            #     print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.10f]' % (
            #         curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
            #         optimizer.param_groups[1]['lr']))
            if curr_iter >= train_args['max_iter']:
                return
            if curr_iter % train_args['val_freq'] == 0:
                validate(val_loader, net, criterion, optimizer, curr_epoch, i + 1, train_args, visualize, val_set)
        curr_epoch += 1


def validate(val_loader, net, criterion, optimizer, epoch, iter_num, train_args, visualize, val_set=None):
    # the following code is written assuming that batch size is 1
    net.eval()
    if train_args['val_save_to_img_file']:
        to_save_dir = os.path.join(ckpt_path, exp_name, '%d_%d' % (epoch, iter_num))
        check_mkdir(to_save_dir)

    outs=[]
    val_loss = AverageMeter()
    gts_all = np.zeros((len(val_loader), int(args['short_size']), int(args['longer_size'])), dtype=int)
    predictions_all = np.zeros((len(val_loader), int(args['short_size']), int(args['longer_size'])), dtype=int)
    #gts_all = np.zeros((len(val_loader), int(args['longer_size']), int(args['short_size'])), dtype=int)
    #predictions_all = np.zeros((len(val_loader), int(args['longer_size']), int(args['short_size'])), dtype=int)

    for vi, data in enumerate(tqdm(val_loader)):
        input, gt, slices_info = data
        assert len(input.size()) == 5 and len(gt.size()) == 4 and len(slices_info.size()) == 3
        input.transpose_(0, 1)
        gt.transpose_(0, 1)
        slices_info.squeeze_(0)
        assert input.size()[3:] == gt.size()[2:]


        #count = torch.zeros(int(args['longer_size']), int(args['short_size'])).cuda()
        #output = torch.zeros(num_classes, int(args['longer_size']), int(args['short_size'])).cuda()
        count = torch.zeros(int(args['short_size']), int(args['longer_size'])).cuda()
        output = torch.zeros(num_classes, int(args['short_size']), int(args['longer_size'])).cuda()
        slice_batch_pixel_size = input.size(1) * input.size(3) * input.size(4)

        for input_slice, gt_slice, info in zip(input, gt, slices_info):
            # print(gt_slice.cpu().numpy())
            # print(np.amax(gt_slice.cpu().numpy()))
            gt_slice = Variable(gt_slice, volatile=True).cuda()
            input_slice = Variable(input_slice, volatile=True).cuda()

            output_slice = net(input_slice)
            assert output_slice.size()[2:] == gt_slice.size()[1:]
            assert output_slice.size()[1] == num_classes
            # print(output_slice.size())
            # print(output.size())
            output[:, info[0]: info[1], info[2]: info[3]] += output_slice[0, :, :info[4], :info[5]].data
            gts_all[vi, info[0]: info[1], info[2]: info[3]] += gt_slice[0, :info[4], :info[5]].data.cpu().numpy()

            count[info[0]: info[1], info[2]: info[3]] += 1

            val_loss.update(criterion(output_slice, gt_slice).data[0], slice_batch_pixel_size)

        output /= count
        outs.append(output.cpu().numpy()[0,:,:])
        np.floor_divide(gts_all[vi, :, :],count.cpu().numpy().astype(int), out=gts_all[vi, :, :])
        #gts_all[vi, :, :] /= count.cpu().numpy().astype(int)
        plt.figure(1, figsize=(11,11), dpi=100)
        plt.subplot(1,3,1)
        plt.imshow(output.cpu().numpy()[0,:,:])
        plt.subplot(1,3,2)
        plt.imshow(output.cpu().numpy()[1, :, :])
        plt.subplot(1, 3, 3)
        plt.imshow(output.cpu().numpy()[2,:,:])
        plt.savefig(os.path.join(to_save_dir, '%d_prediction_imshow.png' % vi), bbox_inches="tight")
        plt.close()
        img_name = os.path.split(val_set.imgs[vi][0])[-1]
        savemat(os.path.join(to_save_dir, img_name[:-4] + ".mat"), {"maps":output.cpu().numpy()})
        #temp_out = output.cpu().numpy()
        #temp_out[1, :, :] *= 1.5
        #print(np.argmax(temp_out, 0))
        #predictions_all[vi, :, :] = np.argmax(temp_out, 0)
        predictions_all[vi, :, :] = output.max(0)[1].squeeze_(0).cpu().numpy()
        # print('validating: %d / %d' % (vi + 1, len(val_loader)))
    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, num_classes)
    if val_loss.avg < train_args['best_record']['val_loss']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['iter'] = iter_num
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
    snapshot_name = 'epoch_%d_iter_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
        epoch, iter_num, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))


    val_visual = []
    for idx, data in enumerate(zip(gts_all, predictions_all)):
        gt_pil = colorize_mask(data[0])
        predictions_pil = colorize_mask(data[1])
        if train_args['val_save_to_img_file']:
            plt.imshow(outs[idx], cmap="jet")
            img_name = os.path.split(val_set.imgs[idx][0])[-1]
            img = cv2.imread(val_set.imgs[idx][0], cv2.IMREAD_COLOR)
            shutil.copy(val_set.imgs[idx][0], os.path.join(to_save_dir, os.path.split(val_set.imgs[idx][0])[-1]))
            predictions_pil.save(os.path.join(to_save_dir, img_name[:-4] + "_prediction.png"))
            gt_pil.save(os.path.join(to_save_dir, img_name[:-4] + "_ground_truth.png"))
            #cv2.imwrite(os.path.join(to_save_dir, img_name[:-4] + "_gt_vs_pred.png"),
            #            show_segmentation_into_original_image(img, data[1]))
            val_visual.extend([visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
    val_visual = torch.stack(val_visual, 0)
    val_visual = utils.make_grid(val_visual, nrow=2, padding=5)
    writer.add_image(snapshot_name, val_visual)

    # print('-----------------------------------------------------------------------------------------------------------')
    # print('[epoch %d], [iter %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
    #     epoch, iter_num, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))
    #
    # print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d], '
    #       '[iter %d]' % (train_args['best_record']['val_loss'], train_args['best_record']['acc'],
    #                      train_args['best_record']['acc_cls'], train_args['best_record']['mean_iu'],
    #                      train_args['best_record']['fwavacc'], train_args['best_record']['epoch'],
    #                      train_args['best_record']['iter']))
    #
    # print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()
