# -*- coding: utf-8 -*-
import datetime
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.adamw import AdamW as TorchAdam

import GLOB as glob
import datasources
import datasets
import models as model_class
from models.utils.initStrategy import InitializeStrategy as InitS
from models.utils.ema import ModelEMA
from comm.base.comm import CommUtils as comm
from comm.misc import ProjectUtils as proj
from comm.pose.process import ProcessUtils as proc
from comm.pose.augment import AugmentUtils as aug
from comm.pose.criteria import AvgCounter, JointsMSELoss, JointsAccuracy
from comm.pose.business import BusinessUtils as bus


def main(mark, params=None):
    args = init_args(params)
    args = proj.project_setting(args, mark)
    args.eval_step = int(args.train_num / (args.batch_size * args.mu))
    logger = glob.get_value('logger')
    logger.print('L1', '=> experiment start, {}'.format(args.experiment))
    logger.print('L1', '=> experiment setting: {}'.format(dict(args._get_kwargs())))

    # region 1. Initialize
    # region 1.1 Data loading
    datasource = datasources.__dict__[args.dataset]()
    args.inp_res, args.out_res = datasource.inp_res, datasource.out_res
    args.num_classes, args.image_type = datasource.kps_num, datasource.image_type
    args.pck_ref, args.pck_thr = datasource.pck_ref, datasource.pck_thr
    targets, labeled_idxs, unlabeled_idxs, valid_idxs = datasource.get_data(args)
    labeled_dataset = datasets.__dict__['PoseDataset'](targets, labeled_idxs, datasource.mean, datasource.std, args, True)
    unlabeled_dataset = datasets.__dict__['PoseDataset_SW'](targets, unlabeled_idxs, datasource.mean, datasource.std, args)
    test_dataset = datasets.__dict__['PoseDataset'](targets, valid_idxs, datasource.mean, datasource.std, args, False)
    # endregion

    # region 1.2 Dataloader initialize
    train_sampler = RandomSampler
    labeled_trainloader = DataLoader(labeled_dataset, sampler=train_sampler(labeled_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = DataLoader(unlabeled_dataset, sampler=train_sampler(unlabeled_dataset), batch_size=args.batch_size*args.mu, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size_infer, num_workers=args.num_workers)
    # endregion

    # region 1.3 Model initialize
    model = model_class.__dict__['PoseModel'](args).to(args.device)
    model_ema = ModelEMA(model, args.ema_decay, args) if args.use_ema else None
    optimizer = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # endregion

    # region 1.4 Hyperparameters initialize
    args.start_epoch = 0
    args.best_acc = [999., -1., -1., -1., -1.]  # [MSE, PCK@0.5, PCK@0.3, PCK@0.2, PCK@0.1]
    args.best_epoch = [0, 0, 0, 0, 0]
    # endregion

    # region 2. Iteration
    logger.print('L1', '=> training start, task: {}@({}/{}), epochs: {}, eval_step: {}, batch-size: {}, use_ema: {}'.format(
        args.dataset, args.num_labeled, args.train_num, args.epochs, args.eval_step, args.batch_size, str(args.use_ema)))
    for epo in range(args.start_epoch, args.epochs):
        epoTM = datetime.datetime.now()
        args.epo = epo

        # region 2.1 model training and validating
        startTM = datetime.datetime.now()
        total_loss, labeled_loss, unlabeled_loss, pl_acc = train(labeled_trainloader, unlabeled_trainloader, model, model_ema, optimizer, args)
        logger.print('L2', 'model training finished...', start=startTM)

        startTM = datetime.datetime.now()
        test_model = model_ema.ema if args.use_ema else model
        predsArray, test_loss, accs = validate(test_loader, test_model, args)
        logger.print('L2', 'model validating finished...', start=startTM)
        # endregion

        # region 2.2 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        for aIdx in range(len(args.pck_thr)):
            is_best = accs[aIdx+1][-1] > args.best_acc[aIdx+1]
            if is_best:
                args.best_epoch[aIdx+1] = epo
                args.best_acc[aIdx+1] = accs[aIdx+1][-1]

        is_best = accs[0][-1] < args.best_acc[0]
        if is_best:
            args.best_epoch[0] = epo
            args.best_acc[0] = accs[0][-1]
        # model storage
        checkpoint = {'current_epoch': args.epo,
                      'best_acc': args.best_acc,
                      'best_epoch': args.best_epoch,
                      'model': args.arch,
                      'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict()}
        if args.use_ema: checkpoint['model_ema_state'] = (
            model_ema.ema.module if hasattr(model_ema.ema, 'module') else model_ema.ema).state_dict()
        comm.ckpt_save(checkpoint, is_best, ckptPath='{}/ckpts'.format(args.basePath))
        logger.print('L2', 'model storage finished...', start=startTM)

        # Log data storage
        log_data = {'total_loss': total_loss,
                    'labeled_loss': labeled_loss,
                    'unlabeled_loss': unlabeled_loss,
                    'pl_acc': pl_acc,
                    'test_loss': test_loss,
                    'best_acc': args.best_acc,
                    'best_epoch': args.best_epoch}
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
        fmtc = '[{}/{} | lr: {} | pl_acc: {}] total_loss: {}, labeled_loss: {}, unlabeled_loss: {}, test_loss: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(optimizer.state_dict()['param_groups'][0]['lr'], '.5f'),
                           format(pl_acc, '.2f'),
                           format(total_loss, '.5f'),
                           format(labeled_loss, '.5f'),
                           format(unlabeled_loss, '.5f'),
                           format(test_loss, '.5f'))
        logger.print('L1', logc)


        fmtc = '[mean][{}/{}] best PCK@{}: {} (epo: {}) | best PCK@{}: {} (epo: {}) | best PCK@{}: {} (epo: {}) | best PCK@{}: {} (epo: {}) | best err: {} (epo: {})'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(args.pck_thr[0], '.1f'),
                           format(args.best_acc[1]*100, '.2f'),
                           format(args.best_epoch[1] + 1, '3d'),
                           format(args.pck_thr[1], '.1f'),
                           format(args.best_acc[2]*100, '.2f'),
                           format(args.best_epoch[2] + 1, '3d'),
                           format(args.pck_thr[2], '.1f'),
                           format(args.best_acc[3]*100, '.2f'),
                           format(args.best_epoch[3] + 1, '3d'),
                           format(args.pck_thr[3], '.1f'),
                           format(args.best_acc[4]*100, '.2f'),
                           format(args.best_epoch[4] + 1, '3d'),
                           format(args.best_acc[0], '.4f'),
                           format(args.best_epoch[0] + 1, '3d'))
        logger.print('L1', logc)

        # Epoch line
        time_interval = logger._interval_format(
            seconds=(datetime.datetime.now() - epoTM).seconds * (args.epochs - (epo + 1)))
        fmtc = '[{}/{} | {}] ---------- ---------- ---------- ---------- ---------- ---------- ----------'
        logc = fmtc.format(format(epo + 1, '3d'), format(args.epochs, '3d'), time_interval)
        logger.print('L1', logc, start=epoTM)
        # endregion
    # endregion


def train(labeled_loader, unlabeled_loader, model, model_ema, optimizer, args):
    # region 1. Preparation
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    total_loss_counter = AvgCounter()
    labeled_loss_counter = AvgCounter()
    unlabeled_loss_counter = AvgCounter()
    pl_acc_counter = AvgCounter()
    pose_criterion = JointsMSELoss().to(args.device)
    # endregion

    # region 2. Training
    model.train()
    for batch_idx in range(args.eval_step):
        log_content = 'epoch: {}-{}: '.format(format(args.epo + 1, '4d'), format(batch_idx + 1, '4d'))
        optimizer.zero_grad()
        # region 2.1 Data organizing
        try:
            inputs_x, targets_x, meta_x = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x, meta_x = next(labeled_iter)

        try:
            # targets_u: Use only when verifying the quality of pseudo-labels
            inputs_u_w, targets_u_w, inputs_u_s, targets_u_s, meta_u = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            # targets_u: Use only when verifying the quality of pseudo-labels
            inputs_u_w, targets_u_w, inputs_u_s, _, meta_u = next(unlabeled_iter)

        batch_size = inputs_x.shape[0]
        inputs = proj.data_interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
        targets_x = targets_x.to(args.device)
        weights_x = meta_x['kps_weight'].to(args.device)
        kps_warpmat_u_w = meta_u['kps_warpmat_w'].to(args.device)
        is_flip_u_w = meta_u['is_flip_w'].to(args.device)
        center_u_w = meta_u['center_w'].to(args.device)
        scale_u_w = meta_u['scale_w'].to(args.device)
        kps_u_test = meta_u['kps_test'].to(args.device)
        kps_warpmat_u_s = meta_u['kps_warpmat_s'].to(args.device)
        is_flip_u_s = meta_u['is_flip_s'].to(args.device)
        # endregion

        # region 2.2 forward
        preds = model(inputs)  # [3*bs, nstack, stream, k, 64, 64], [3*bs, nstack, stream, 25, 64, 64]
        logits_x, logits_u_w, logits_u_s = proj.data_de_interleave_group(preds, batch_size, args)
        # endregion

        # region 2.3 supervised learning loss
        labeled_loss, kps_count = pose_criterion(logits_x, targets_x, weights_x)
        labeled_loss_counter.update(labeled_loss.item(), kps_count)
        log_content += ' loss_x: {}'.format(format(labeled_loss.item(), '.5f'))
        # endregion

        # region 2.4 get ensemble prediction
        targets_hm_ens = aug.affine_back(logits_u_w[:, -1].clone(), kps_warpmat_u_w, is_flip_u_w)
        targets_pred_ens, _ = proc.kps_from_heatmap(targets_hm_ens.cpu(), center_u_w, scale_u_w, [args.out_res, args.out_res])
        targets_ens_errs, _ = JointsAccuracy.pck(targets_pred_ens, kps_u_test.float().cpu(), args.pck_ref, args.pck_thr[0])
        pl_acc_counter.update(targets_ens_errs[-1])
        log_content += ' | pl_err: {}'.format(format(targets_ens_errs[-1], '.3f'))
        # endregion

        # region 2.5 weak supervises strong
        if args.lambda_ens > 0:
            logits_u_s_hm_back = aug.affine_back(logits_u_s[:, -1].clone(), kps_warpmat_u_s, is_flip_u_s)
            unlabeled_loss, kps_count = pose_criterion(logits_u_s_hm_back, targets_hm_ens.detach())
            unlabeled_loss_counter.update(unlabeled_loss.item(), kps_count)
            log_content += ' | loss_ens: {}'.format(format(unlabeled_loss.item(), '.3f'))
        else:
            unlabeled_loss = torch.tensor(0.).to(args.device)
            unlabeled_loss_counter.update(0., 1)
        # endregion

        # region 2.8 calculate total loss & update model
        if args.epo < args.ensemble_warmup:
            loss = args.lambda_la * labeled_loss
        else:
            loss = args.lambda_la * labeled_loss + args.lambda_ens * unlabeled_loss
        total_loss_counter.update(loss.item())
        loss.backward()
        optimizer.step()
        if args.use_ema: model_ema.update(model)
        # endregion
        # if (batch_idx + 1) % int(args.eval_step/2) == 0 or (batch_idx + 1) == args.eval_step: print(log_content)

    # region 3. calculate records
    total_loss_val = total_loss_counter.avg
    labeled_loss_val = labeled_loss_counter.avg
    unlabeled_loss_val = unlabeled_loss_counter.avg
    pl_acc_val = pl_acc_counter.avg.item()
    # endregion
    return total_loss_val, labeled_loss_val, unlabeled_loss_val, pl_acc_val


def validate(test_loader, model, args):
    # region 1. Preparation
    test_loss_counter = AvgCounter()  # mse, KCP1, 2, 3, 4
    acc_counters = torch.zeros(len(args.best_acc), args.num_classes + 1)
    predsArray = []

    pose_criterion = JointsMSELoss().to(args.device)
    test_iter = iter(test_loader)
    eval_step = int(args.train_num/args.batch_size_infer)
    model.eval()
    # endregion

    # region 2. Validating
    with torch.no_grad():
        for batch_idx in range(eval_step):
            # region 2.1 Data organizing
            try:
                inputs, targets, meta = next(test_iter)
            except:
                test_iter = iter(test_loader)
                inputs, targets, meta = next(test_iter)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            weights = meta['kps_weight'].to(args.device)
            bs, k, _, _ = targets.shape
            # endregion

            # region 2.2 forward
            logits = model(inputs)
            # endregion

            # region 2.3 calculate loss & accuracy
            # cal test loss
            test_loss, kps_count = pose_criterion(logits, targets, weights)
            test_loss_counter.update(test_loss.item(), kps_count)

            # get predictions and cal accuracy
            preds, _ = proc.kps_from_heatmap(logits[:, -1].cpu(), meta["center"], meta["scale"], [args.out_res, args.out_res])
            predsArray += preds.clone().data.numpy().tolist()
            for pck_thr_idx in range(len(args.pck_thr)):
                errs, accs = JointsAccuracy.pck(preds, meta['kps'], args.pck_ref, args.pck_thr[pck_thr_idx])
                for kIdx in range(k+1):
                    if pck_thr_idx == 0:
                        acc_counters[0][kIdx] = errs[kIdx].item()
                    else:
                        acc_counters[pck_thr_idx][kIdx] = accs[kIdx].item()
    return predsArray, test_loss_counter.avg, acc_counters.tolist()


def init_args(params=None):
    parser = argparse.ArgumentParser(description='FixMatch Training')

    # Model Setting
    parser.add_argument('--arch', default='Hourglass', type=str, choices=['Hourglass', 'Hourglass_MS'], help='model name')
    parser.add_argument('--stack-num', default=2, type=int, help='number of stack')
    parser.add_argument('--use-ema', default='False')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')

    # Mulitple Stream
    parser.add_argument('--ensemble-warmup', default=5, type=int, help='number of epoch')
    parser.add_argument('--lambda-ens', default=1, type=float, help='coefficient of ensemble prediction loss')

    # Dataset setting
    parser.add_argument('--dataset', default="Mouse", choices=["Mouse", "FLIC", "LSP"], help='dataset name')
    parser.add_argument('--train-num', default=100, type=int, help='number of total training data')
    parser.add_argument('--num-labeled', type=int, default=30, help='number of labeled data')
    parser.add_argument('--valid-num', default=500, type=int, help='number of validating data')

    # Weak Data augment (default)
    parser.add_argument("--use-flip", default="True", help="whether add flip augment")
    parser.add_argument("--sf", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rf", default=30.0, type=float, help="rotation factor")

    # Strong Data augment (for student)
    # 大rf_s时，不能使用False
    parser.add_argument("--use-flip-s", default="True", help="whether add flip augment")
    parser.add_argument("--sf-s", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rf-s", default=45.0, type=float, help="rotation factor")

    # Training strategy
    parser.add_argument('--epochs', default=200, type=int, help='number of total steps to run')
    parser.add_argument('--batch-size', default=4, type=int, help='train batchsize')
    parser.add_argument('--batch-size-infer', default=16, type=int, help='train batchsize')
    parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--expand-labels', default='True', help='expand labels to fit eval steps')
    parser.add_argument("--lr", default=2.5e-4, type=float, help="initial learning rate")
    parser.add_argument("--power", default=0.9, type=float, help="power for learning rate decay")
    parser.add_argument("--wd", default=0, type=float, help="weight decay (default: 0)")
    parser.add_argument('--lambda-la', default=10, type=float, help='coefficient of unlabeled loss')

    # misc
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--seed', default=1388, type=int, help='random seed')
    parser.add_argument('--debug', default='True', help='do debug operation')

    # params set-up
    args = proj.project_args_setup(parser.parse_args(), params)
    return args


if __name__ == '__main__':
    main('DualPose')