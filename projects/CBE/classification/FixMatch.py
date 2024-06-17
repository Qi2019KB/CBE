# -*- coding: utf-8 -*-
import GLOB as glob
import datetime
import argparse
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import datasources
import datasets
import models as model_class
from models.utils.ema import ModelEMA
from comm.base.comm import CommUtils as comm
from comm.misc import ProjectUtils as proj
from comm.schedule import ScheduleUtils as schedule
from comm.classification.criteria import AvgCounter, ClassAccuracy
from comm.classification.business import BusinessUtils as bus


def main(mark, params=None):
    args = init_args(params)
    args = proj.project_setting(args, mark)
    logger = glob.get_value('logger')
    logger.print('L1', '=> experiment start, {}'.format(args.experiment))
    logger.print('L1', '=> experiment setting: {}'.format(dict(args._get_kwargs())))

    # region 1. Initialize
    # region 1.1 Data loading
    datasource = datasources.__dict__[args.dataset]()
    labeled_idx, unlabeled_idx = datasource.get_data(args)
    args.num_classes = datasource.num_classes
    args.name_classes = datasource.name_classes
    labeled_dataset, unlabeled_dataset, test_dataset = datasets.__dict__[args.dataset](
        labeled_idx, unlabeled_idx, datasource.root, datasource.mean, datasource.std)
    # endregion

    # region 1.2 Dataloader initialize
    train_sampler = RandomSampler
    labeled_loader = DataLoader(labeled_dataset,
                                sampler=train_sampler(labeled_dataset),
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset,
                                  sampler=train_sampler(unlabeled_dataset),
                                  batch_size=args.batch_size * args.mu,
                                  num_workers=args.num_workers,
                                  drop_last=True)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    # endregion

    # region 1.3 Model initialize
    model = model_class.__dict__['ClassModel'](args)
    model_ema = ModelEMA(model, args.ema_decay, args) if args.use_ema else None

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=args.power, nesterov=args.nesterov)
    scheduler = schedule.get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    # endregion

    # region 1.4 Hyperparameters initialize
    args.start_epoch = 0
    args.best_acc = -1.
    args.best_epoch = 0
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    # endregion
    # endregion

    # region 2. Iteration
    logger.print('L1', '=> task: {}@{}, epochs: {}, eval-step: {}, batch-size: {}'.format(
        args.dataset, args.num_labeled, args.epochs, args.eval_step, args.batch_size))
    for epo in range(args.start_epoch, args.epochs):
        epoTM = datetime.datetime.now()
        args.epo = epo

        # region 2.1 model training and validating
        startTM = datetime.datetime.now()
        total_loss, labeled_loss, unlabeled_loss, mask, pl_acc = train(
            labeled_loader, unlabeled_loader, model, model_ema, optimizer, scheduler, args)
        logger.print('L2', 'model training finished...', start=startTM)

        startTM = datetime.datetime.now()
        test_model = model_ema.ema if args.use_ema else model
        predsArray, test_loss, acc_t1, acc_t5 = validate(test_loader, test_model, args)
        logger.print('L2', 'model validating finished...', start=startTM)
        # endregion

        # region 2.2 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = acc_t1 > args.best_acc
        if is_best:
            args.best_epoch = epo
            args.best_acc = acc_t1
        # model storage
        checkpoint = {'current_epoch': args.epo,
                      'best_acc': args.best_acc,
                      'best_epoch': args.best_epoch,
                      'model': args.arch,
                      'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'scheduler_state': scheduler.state_dict()}

        if args.use_ema: checkpoint['model_ema_state'] = (
            model_ema.ema.module if hasattr(model_ema.ema, 'module') else model_ema.ema).state_dict()
        comm.ckpt_save(checkpoint, is_best, ckptPath='{}/ckpts'.format(args.basePath))
        logger.print('L2', 'model storage finished...', start=startTM)

        # Log data storage
        log_data = {'total_loss': total_loss,
                    'labeled_loss': labeled_loss,
                    'unlabeled_loss': unlabeled_loss,
                    'mask': mask,
                    'pl_acc': pl_acc,
                    'test_loss': test_loss,
                    'acc_t1': acc_t1,
                    'acc_t5': acc_t5}
        log_save_path = '{}/logs/logData/logData_{}.json'.format(args.basePath, epo + 1)
        comm.json_save(log_data, log_save_path, isCover=True)

        # Pseudo-labels data storage
        # pseudo_data = {'predsArray': predsArray}
        # pseudo_save_path = '{}/logs/pseudoData/pseudoData_{}.json'.format(args.basePath, epo + 1)
        # comm.json_save(pseudo_data, pseudo_save_path, isCover=True)
        # logger.print('L2', 'log storage finished...', start=startTM)
        # endregion

        # region 2.3 output result
        # Training performance
        fmtc = '[{}/{} | mask: {}, pl_acc: {}] loss: {}, loss_x: {}, loss_u: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(mask, '.5f'),
                           format(pl_acc, '.5f'),
                           format(total_loss, '.3f'),
                           format(labeled_loss, '.5f'),
                           format(unlabeled_loss, '.3f'))
        logger.print('L1', logc)

        # Validating performance
        fmtc = '[{}/{}] best acc: {} (epo: {}) | test_loss: {} | top1: {}, top5: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(args.best_acc, '.5f'),
                           format(args.best_epoch + 1, '3d'),
                           format(test_loss, '.3f'),
                           format(acc_t1, '.2f'),
                           format(acc_t5, '.2f'))
        logger.print('L1', logc)

        # Epoch line
        time_interval = logger._interval_format(
            seconds=(datetime.datetime.now() - epoTM).seconds * (args.epochs - (epo + 1)))
        fmtc = '[{}/{} | {}] ---------- ---------- ---------- ---------- ---------- ---------- ----------'
        logc = fmtc.format(format(epo + 1, '3d'), format(args.epochs, '3d'), time_interval)
        logger.print('L1', logc, start=epoTM)
        # endregion
    # endregion


def train(labeled_loader, unlabeled_loader, model, model_ema, optimizer, scheduler, args):
    # region 1. Preparation
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    total_loss_counter = AvgCounter()
    labeled_loss_counter = AvgCounter()
    unlabeled_loss_counter = AvgCounter()
    mask_probs = AvgCounter()
    pl_acc_counter = AvgCounter()
    stat_pl_preds, stat_pl_targets, stat_pl_masks = None, None, None
    # endregion

    # region 2. Training
    model.train()
    for batch_idx in range(args.eval_step):
        log_content = 'epoch: {}-{}: '.format(format(args.epo + 1, '4d'), format(batch_idx + 1, '4d'))
        optimizer.zero_grad()
        # region 2.1 Data organizing
        try:
            inputs_x, targets_x = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = next(labeled_iter)

        try:
            # targets_u: Use only when verifying the quality of pseudo-labels
            (inputs_u_w, inputs_u_s), targets_u = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            # targets_u: Use only when verifying the quality of pseudo-labels
            (inputs_u_w, inputs_u_s), targets_u = next(unlabeled_iter)

        batch_size = inputs_x.shape[0]
        inputs = proj.data_interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
        targets_x = targets_x.to(args.device)
        targets_u = targets_u.to(args.device)
        # endregion

        # region 2.2 forward
        logits = model(inputs)
        logits_x, logits_u_w, logits_u_s = proj.data_de_interleave_group(logits, batch_size, args)
        del logits
        # endregion

        # region 2.3 supervised learning loss
        labeled_loss = F.cross_entropy(logits_x, targets_x.long().to(args.device), reduction='mean')
        labeled_loss_counter.update(labeled_loss.item())
        log_content += 'loss_x: {}'.format(format(labeled_loss.item(), '.5f'))
        # endregion

        # region 2.4 get pseudo-labels
        pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
        max_probs, pseudo_label_u = torch.max(pseudo_label, dim=-1)

        mask = max_probs.ge(args.threshold).float()
        mask_probs.update(mask.mean().item())

        pl_acc = bus.target_verify(pseudo_label_u, targets_u, mask)
        pl_acc_counter.update(pl_acc.item())

        mask_len = len([item for item in mask if item > 0])
        mask_count = logits_u_s.shape[0]
        # endregion

        # region 2.5 unsupervised learning loss
        unlabeled_loss = (F.cross_entropy(logits_u_s, pseudo_label_u, reduction='none') * mask).mean()
        unlabeled_loss_counter.update(unlabeled_loss.item())
        log_content += ' | loss_u: {}'.format(format(unlabeled_loss.item(), '.3f'))

        log_content += ' [mask: {} ({}/{}); pl acc: {}]'.format(
            format(mask.mean().item(), '.2f'),
            format(mask_len, '4d'),
            format(mask_count, '4d'),
            format(pl_acc.item(), '.2f'))
        # endregion

        # region 2.6 calculate total loss & update model
        total_loss = labeled_loss + args.lambda_fm * unlabeled_loss
        total_loss_counter.update(total_loss.item())
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        if args.use_ema: model_ema.update(model)
        # endregion
        # print(log_content)

        # region 2.7 visualization
        if args.debug:
            # 2.7.1 pseudo-labels distribution visualization
            stat_pl_preds = pseudo_label_u if stat_pl_preds is None else torch.cat([stat_pl_preds, pseudo_label_u])
            stat_pl_targets = targets_u if stat_pl_targets is None else torch.cat([stat_pl_targets, targets_u])
            stat_pl_masks = mask if stat_pl_masks is None else torch.cat([stat_pl_masks, mask])

            if ((batch_idx + 1) >= 200 and (batch_idx + 1) % 200 == 0) or (batch_idx + 1) == args.eval_step:
                dist_box_preds = bus.target_statistic_train(args, stat_pl_preds, stat_pl_targets)
                save_path = '{}/{}/distribution_visualization/prediction/e{}_b{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1)
                proj.distribution_visualize(dist_box_preds, save_path, args.name_classes)

                dist_box_preds_sel = bus.target_statistic_train(args, stat_pl_preds, stat_pl_targets, stat_pl_masks)
                save_path = '{}/{}/distribution_visualization/pseudo_labels/e{}_b{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1)
                proj.distribution_visualize(dist_box_preds_sel, save_path, args.name_classes)
                stat_pl_preds, stat_pl_targets, stat_pl_masks = None, None, None
            # endregion
        # endregion
    # endregion

    return total_loss_counter.avg, labeled_loss_counter.avg, unlabeled_loss_counter.avg, mask_probs.avg, pl_acc_counter.avg


def validate(test_loader, model, args):
    # region 1. Preparation
    test_loss_counter = AvgCounter()
    top1_counter = AvgCounter()
    top5_counter = AvgCounter()
    predsArray = []
    # endregion

    # region 2. Validating
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # region 2.1 data organize
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # endregion

            # region 2.2 forward
            outputs = model(inputs)
            # endregion

            # region 2.3 calculate loss & accuracy
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = ClassAccuracy.accuracy(outputs, targets, topk=(1, 5))
            test_loss_counter.update(loss.item(), inputs.shape[0])
            top1_counter.update(prec1.item(), inputs.shape[0])
            top5_counter.update(prec5.item(), inputs.shape[0])
            # prediction

            # region 2.4 get prediction
            _, preds_ema = torch.max(outputs, -1)
            predsArray += preds_ema.clone().cpu().data.numpy().tolist()
            # prediction
    # endregion
    return predsArray, test_loss_counter.avg, top1_counter.avg, top5_counter.avg


def init_args(params=None):
    parser = argparse.ArgumentParser(description='FixMatch Training')

    # Model Setting
    parser.add_argument('--arch', default='WideResNet', type=str, choices=['WideResNet', 'ResNeXt'], help='model name')
    parser.add_argument('--use-ema', default='True')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')

    # Dataset setting
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN'], help='dataset name')
    parser.add_argument('--train-num', default=50000, type=int, help='number of total training data')
    parser.add_argument('--num-labeled', type=int, default=40, help='number of labeled data')
    parser.add_argument('--valid-num', default=10000, type=int, help='number of validating data')

    # Training strategy
    parser.add_argument('--total-steps', default=1024*1024, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int, help='number of eval steps to run')  # 1024
    parser.add_argument('--batch-size', default=32, type=int, help='train batchsize')  # 32
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--expand-labels', default='True', help='expand labels to fit eval steps')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument("--power", default=0.9, type=float, help="power for learning rate decay")
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--lambda-fm', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')  # 0.95

    # misc
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--seed', default=1388, type=int, help='random seed')
    parser.add_argument('--debug', default='True')

    # params set-up
    args = proj.project_args_setup(parser.parse_args(), params)
    return args


if __name__ == '__main__':
    main('FixMatch', {'gpu_id': 0, 'debug': True, 'dataset': 'SVHN', 'num_labeled': 40})  # , 'total_steps': 32*50, 'eval_step': 32
