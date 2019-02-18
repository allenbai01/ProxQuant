## Binary neural nets via prox operations
import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import models
import pdb
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from reg import *
from writer import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--save_all', action='store_true',
                    help='save model at every epoch')
parser.add_argument('--tb_dir', type=str, default=None,
                    help='TensorBoard directory')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--binary_reg', default=0.0, type=float,
                    help='Binary regularization strength')
parser.add_argument('--reg_rate', default=0.0, type=float,
                    help='Regularization rate')
parser.add_argument('--adjust_reg', action='store_true',
                    help='Adjust regularization based on learning rate decay')
parser.add_argument('--projection_mode', default=None, type=str,
                    help='Projection / rounding mode')
parser.add_argument('--freeze_epoch', default=-1, type=int,
                    help='Epoch to freeze quantization')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--binarize', action='store_true',
                    help='Load an existing model and binarize')
parser.add_argument('--binary_regime', action='store_true',
                    help='Use alternative stepsize regime (for binary training)')
parser.add_argument('--ttq_regime', action='store_true',
                    help='Use alternative stepsize regime (for ttq)')
parser.add_argument('--no_adjust', action='store_true',
                    help='Will not adjust learning rate')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    writer = TensorboardWriter(args.tb_dir)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    model.cuda(device=args.gpus[0])
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            # results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            # args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            best_prec1 = best_prec1.cuda(args.gpus[0])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Adjust batchnorm layers if in stochastic binarization mode
    if args.projection_mode == 'stoch_bin':
        adjust_bn(model)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})

    # Adjust stepsize regime for specific optimizers
    if args.binary_regime:
        regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-2},
            81: {'lr': 1e-3},
            122: {'lr': 1e-4},
        }
    elif args.ttq_regime:
        regime = {
            0: {'optimizer': 'SGD', 'lr': 0.1,
                'momentum': 0.9, 'weight_decay': 2e-4},
            80: {'lr': 1e-2},
            120: {'lr': 1e-3},
            300: {'lr': 1e-4}
        }
    elif args.optimizer == 'Adam':
        regime = {
            0 : {'optimizer': 'Adam', 'lr': args.lr},
        }
    elif args.projection_mode != None:
        # Remove weight decay when using SGD, and reset momentum
        regime[0]['weight_decay'] = 0.0
        regime[0]['momentum'] = args.momentum

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr)
    logging.info('training regime: %s', regime)

    bin_op = BinOp(model,
                   if_binary=if_binary_tern if args.projection_mode in [
                       'prox_ternary', 'ttq'] else if_binary,
                   ttq=(args.projection_mode == 'ttq'))

    # Optionally freeze before training
    if args.freeze_epoch == 1:
        bin_op.quantize(mode='binary_freeze')
        args.projection_mode = None
    # Loop over epochs
    try:
        for epoch in range(args.start_epoch, args.epochs):
            if not(args.no_adjust):
                optimizer = adjust_optimizer(optimizer, epoch, regime)

            # train for one epoch
            # Adjust binary regression mode if non-lazy projection
            if args.projection_mode in ['prox', 'prox_median', 'prox_ternary']:
                br = args.reg_rate * epoch
            else:
                br = args.binary_reg
            # Adjust binary reg according to learning rate
            if args.adjust_reg:
                curr_lr = optimizer.param_groups[0]['lr']
                br *= args.lr / curr_lr
                
            train_loss, train_prec1, train_prec5 = train(
                train_loader, model, criterion, epoch, optimizer,
                br=br, bin_op=bin_op, projection_mode=args.projection_mode)

            # evaluate on validation set
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, criterion, epoch,
                br=br, bin_op=bin_op, projection_mode=args.projection_mode)

            # evaluate binarized model
            val_loss_bin, val_prec1_bin, val_prec5_bin = validate(
                val_loader, model, criterion, epoch,
                br=br, bin_op=bin_op, projection_mode=args.projection_mode,
                binarize=True)

            # remember best prec@1 and save checkpoint
            # Look at prec@1 for either binarized model or original model
            if args.binary_reg > 1e-10 or args.reg_rate > 1e-10:
                is_best = val_prec1_bin > best_prec1
                best_prec1 = max(val_prec1_bin, best_prec1)
            else:
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'config': args.model_config,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'regime': regime
            }, is_best, path=save_path, save_all=args.save_all)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Training Prec@5 {train_prec5:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1,
                                 train_prec5=train_prec5, val_prec5=val_prec5))
            
            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                        train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                        train_error5=100 - train_prec5, val_error5=100 - val_prec5)
            #results.plot(x='epoch', y=['train_loss', 'val_loss'],
            #             title='Loss', ylabel='loss')
            #results.plot(x='epoch', y=['train_error1', 'val_error1'],
            #             title='Error@1', ylabel='error %')
            #results.plot(x='epoch', y=['train_error5', 'val_error5'],
            #             title='Error@5', ylabel='error %')
            results.save()
            result_dict = {'train_loss': train_loss, 'val_loss': val_loss,
                           'train_error1': 100 - train_prec1, 'val_error1': 100 - val_prec1,
                           'train_error5': 100 - train_prec5, 'val_error5': 100 - val_prec5,
                           'val_loss_bin': val_loss_bin,
                           'val_error1_bin': 100 - val_prec1_bin,
                           'val_error5_bin': 100 - val_prec5_bin}
            writer.write(result_dict, epoch+1)
            writer.write(binary_levels(model), epoch+1)
            
            # Compute general quantization error
            mode = 'ternary' if args.projection_mode == 'prox_ternary' else 'deterministic'
            writer.write(bin_op.quantize_error(mode=mode), epoch+1)
            writer.write(sign_changes(bin_op), epoch+1)
            if bin_op.ttq:
                writer.write(bin_op.ternary_vals, epoch+1)
            # writer.export()

            # Optionally freeze the binarization at a given epoch
            if args.freeze_epoch > 0 and epoch+1 == args.freeze_epoch:
                if args.projection_mode in ['prox', 'lazy']:
                    bin_op.quantize(mode='binary_freeze')
                elif args.projection_mode == 'prox_ternary':
                    bin_op.quantize(mode='ternary_freeze')
                args.projection_mode = None
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Wrap up
    writer.close()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None,
            br=0.0, bin_op=None, projection_mode=None, binarize=False):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # Binarize or prox-opearate the model if in eval mode
    if not(training):
        bin_op.save_params()
        if binarize:
            # bin_op.binarize()
            if projection_mode == 'prox_median':
                bin_op.quantize('median')
            elif projection_mode == 'prox_ternary':
                bin_op.quantize('ternary')
            elif projection_mode in ['prox', 'lazy']:
                bin_op.quantize('deterministic')
            elif projection_mode == 'ttq':
                bin_op.quantize('ttq')
        elif projection_mode == 'lazy':
            bin_op.prox_operator(br, 'binary')
    
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()
            # target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # Binarize if projection mode is {lazy, stochastic bin} and in training
        if training:
            if projection_mode == 'lazy':
                bin_op.save_params()
                bin_op.prox_operator(br, 'binary')
            elif projection_mode == 'ttq':
                bin_op.save_params()
                bin_op.quantize('ttq')
            elif projection_mode == 'stoch_bin':
                bin_op.save_params()
                bin_op.binarize(mode='stochastic')
                # pdb.set_trace()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # loss_reg = loss + br * binary_reg(model)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # copy parameters according to quantization modes
            if projection_mode in ['lazy', 'stoch_bin']:
                bin_op.restore()
                optimizer.step()
                bin_op.clip()
            elif projection_mode == 'ttq':
                bin_op.restore()
                optimizer.step()
                step_ternary_vals(bin_op, optimizer)
            elif projection_mode in ['prox', 'prox_median', 'prox_ternary']:
                optimizer.step()
                curr_lr = optimizer.param_groups[0]['lr']
                if projection_mode == 'prox':
                    bin_op.prox_operator(curr_lr * br, 'binary')
                elif projection_mode == 'prox_median':
                    bin_op.prox_operator(curr_lr * br, 'median')
                elif projection_mode == 'prox_ternary':
                    bin_op.prox_operator(curr_lr * br, 'ternary')
                bin_op.clip()
            else:
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    if not(training):
        bin_op.restore()
    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer,
          br=0.0, bin_op=None, projection_mode=None):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer,
                   br=br, bin_op=bin_op, projection_mode=projection_mode)


def validate(data_loader, model, criterion, epoch,
             br=0.0, bin_op=None, projection_mode=None,
             binarize=False):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None,
                   br=br, bin_op=bin_op, projection_mode=projection_mode,
                   binarize=binarize)


if __name__ == '__main__':
    main()
