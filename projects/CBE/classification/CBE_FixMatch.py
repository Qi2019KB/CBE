# -*- coding: utf-8 -*-
import GLOB as glob
import numpy as np
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
from models.utils.initStrategy import InitializeStrategy as InitS
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
        labeled_idx, unlabeled_idx, datasource.root, datasource.mean, datasource.std
    )
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
                                  batch_size=args.batch_size*args.mu,
                                  num_workers=args.num_workers,
                                  drop_last=True)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    # endregion

    # region 1.3 Model initialize
    model = model_class.__dict__['ClassModel'](args)
    InitS.ms_fc_initialize(model)
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
        total_loss, labeled_losses, ensemble_losses, fd_loss, mc_losses, mask, pl_acc, sim_sum, wrong_sum, sim_rate = train(
            labeled_loader, unlabeled_loader, model, model_ema, optimizer, scheduler, args)
        logger.print('L2', 'model training finished...', start=startTM)

        startTM = datetime.datetime.now()
        test_model = model_ema.ema if args.use_ema else model
        predsArray, test_losses, t1s, t5s = validate(test_loader, test_model, args)
        logger.print('L2', 'model validating finished...', start=startTM)
        # endregion

        # region 2.2 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = t1s[-1] > args.best_acc
        if is_best:
            args.best_epoch = epo
            args.best_acc = t1s[-1]
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
                    'labeled_losses': labeled_losses,
                    'ensemble_losses': ensemble_losses,
                    'pl_similarity': sim_rate,
                    'pl_sim_num': sim_sum,
                    'pl_wrong_num': wrong_sum,
                    'mask': mask,
                    'pl_acc': pl_acc,
                    'fd_loss': fd_loss,
                    'mc_losses': mc_losses,
                    'test_losses': test_losses,
                    't1s': t1s,
                    't5s': t5s}
        comm.json_save(log_data, '{}/logs/logData/logData_{}.json'.format(args.basePath, epo+1), isCover=True)

        # Pseudo-labels data storage
        # pseudo_data = {'predsArray': predsArray}
        # comm.json_save(pseudo_data, '{}/logs/pseudoData/pseudoData_{}.json'.format(args.basePath, epo+1), isCover=True)
        # logger.print('L2', 'log storage finished...', start=startTM)
        # endregion

        # region 2.3 output result
        # Training performance
        fmtc = '[{}/{} | pl_mask: {}, pl_acc: {} | pl_sim: {} ({}/ {})] total_loss: {}, x_loss: {}, ens_loss: {}, fd_loss: {}, mc_loss: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(mask, '.5f'),
                           format(pl_acc, '.5f'),
                           format(sim_rate, '.3f'),
                           format(sim_sum, '5d'),
                           format(wrong_sum, '5d'),
                           format(total_loss, '.3f'),
                           format(labeled_losses[-1], '.5f'),
                           format(ensemble_losses[-1], '.3f'),
                           format(fd_loss, '.8f'),
                           format(mc_losses[-1], '.8f'))
        logger.print('L1', logc)

        for stIdx in range(args.stream_num):
            fmtc = '[{}/{} | ms{}] x_loss: {}, ens_loss: {}, mc_loss: {}'
            logc = fmtc.format(format(epo + 1, '3d'),
                               format(args.epochs, '3d'),
                               stIdx+1,
                               format(labeled_losses[stIdx], '.5f'),
                               format(ensemble_losses[stIdx], '.3f'),
                               format(mc_losses[stIdx], '.8f'))
            logger.print('L2', logc)

        # Validating performance
        fmtc = '[{}/{} | count_thr: {}, score_thr: {}] best acc: {} (epo: {}) | test_loss: {} | top1: {}, top5: {}'
        logc = fmtc.format(format(epo + 1, '3d'),
                           format(args.epochs, '3d'),
                           format(args.count_thr, '1d'),
                           format(args.score_thr, '.3f'),
                           format(args.best_acc, '.2f'),
                           format(args.best_epoch + 1, '3d'),
                           format(test_losses[-1], '.3f'),
                           format(t1s[-1], '.2f'),
                           format(t5s[-1], '.2f'))
        logger.print('L1', logc)

        for stIdx in range(args.stream_num):
            fmtc = '[{}/{} | ms{}] test_loss: {} | top1: {}, top5: {}'
            logc = fmtc.format(format(epo + 1, '3d'),
                               format(args.epochs, '3d'),
                               stIdx+1,
                               format(test_losses[stIdx], '.3f'),
                               format(t1s[stIdx], '.2f'),
                               format(t5s[stIdx], '.2f'))
            logger.print('L2', logc)

        # Epoch line
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
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
    labeled_loss_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    ensemble_loss_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    mask_probs_counter = AvgCounter()
    pl_acc_counter = AvgCounter()
    fd_loss_counter = AvgCounter()
    mc_loss_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    stat_pl_preds = [None for _ in range(args.stream_num + 1)]
    stat_pl_masks = [None for _ in range(args.stream_num + 1)]
    stat_pl_targets = None
    sim_num_counter = AvgCounter()
    wrong_num_counter = AvgCounter()
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
        inputs = proj.data_interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
        targets_x = targets_x.to(args.device)
        targets_u = targets_u.to(args.device)
        # endregion

        # region 2.2 forward
        ms_preds, ms_fs_p = model(inputs)
        ms_logits_x, ms_logits_u_w, ms_logits_u_s, ms_features = [], [], [], []
        for stIdx in range(args.stream_num):
            logits_x, logits_u_w, logits_u_s = proj.data_de_interleave_group(ms_preds[stIdx], batch_size, args)
            ms_logits_x.append(logits_x)
            ms_logits_u_w.append(logits_u_w)
            ms_logits_u_s.append(logits_u_s)
            _, _, fs_p_s = proj.data_de_interleave_group(ms_fs_p[stIdx], batch_size, args)
            ms_features.append(fs_p_s)
        del ms_preds, ms_fs_p
        ms_logits_x = torch.stack(ms_logits_x, dim=0)
        ms_logits_u_w = torch.stack(ms_logits_u_w, dim=0)
        ms_logits_u_s = torch.stack(ms_logits_u_s, dim=0)
        ms_features = torch.stack(ms_features, dim=0)
        # endregion

        # region 2.3 supervised learning loss
        blank_stIdx, unblank_sum = batch_idx % args.stream_num, 0
        blank_idxs = [blank_stIdx, blank_stIdx+1][0:args.blank_num]
        labeled_loss_sum = torch.tensor(0.).to(args.device)
        for stIdx in range(args.stream_num):
            labeled_loss_val = F.cross_entropy(ms_logits_x[stIdx], targets_x.long().to(args.device), reduction='none').mean()
            labeled_loss_counters[stIdx].update(labeled_loss_val.item())
            if stIdx not in blank_idxs:
                labeled_loss_sum += labeled_loss_val
                unblank_sum += 1
        labeled_loss = labeled_loss_sum / unblank_sum
        labeled_loss_counters[-1].update(labeled_loss.item(), unblank_sum)
        log_content += ' loss_x: {}'.format(format(labeled_loss.item(), '.5f'))
        # endregion

        # region 2.4 get ensemble prediction
        sim_num, wrong_num = bus.prediction_similarity(ms_logits_u_w, targets_u)
        sim_num_counter.update(sim_num)
        wrong_num_counter.update(wrong_num)

        targets_ens, _, masks, ms_max_idx, ms_masks = bus.target_ensemble_mask(ms_logits_u_w, args)
        pl_acc = bus.target_verify(targets_ens, targets_u, masks)
        pl_acc_counter.update(pl_acc.item())
        masks_len, masks_count = len([item for item in masks if item > 0]), targets_ens.shape[0]
        mask_probs_counter.update(masks.mean().item())
        log_content += ' | sim: {}, wrong: {}'.format(format(sim_num, '2d'), format(wrong_num, '2d'))
        # endregion

        # region 2.5 ensemble prediction constraint
        if args.lambda_ens > 0:
            blank_stIdx, unblank_sum = batch_idx % args.stream_num, 0
            blank_idxs = [blank_stIdx, blank_stIdx+1][0:args.blank_num]
            ensemble_loss_sum = torch.tensor(0.).to(args.device)
            for stIdx in range(args.stream_num):
                ensemble_loss_val = (F.cross_entropy(ms_logits_u_s[stIdx], targets_ens.detach(), reduction='none', ignore_index=-1) * masks).mean()
                ensemble_loss_counters[stIdx].update(ensemble_loss_val.item())
                if stIdx not in blank_idxs:
                    ensemble_loss_sum += ensemble_loss_val
                    unblank_sum += 1
            ensemble_loss = ensemble_loss_sum / unblank_sum
            ensemble_loss_counters[-1].update(ensemble_loss.item(), unblank_sum)
            log_content += ' | loss_ens: {}'.format(format(ensemble_loss.item(), '.3f'))

            log_content += ' [mask: {} ({}/{}); pl acc: {}]'.format(
                format(masks.mean().item(), '.2f'),
                format(masks_len, '4d'),
                format(masks_count, '4d'),
                format(pl_acc.item(), '.2f'))
        else:
            ensemble_loss = torch.tensor(0.).to(args.device)
            for counter in ensemble_loss_counters: counter.update(0., 1)
        # endregion

        # region 2.6 multi-view features decorrelation loss
        if args.lambda_fd > 0:
            fd_loss_sum, fd_loss_count = torch.tensor(0.).to(args.device), 0
            for i in range(args.stream_num):
                j = i + 1 if i + 1 < args.stream_num else 0
                covar_val, covar_num = bus.corrcoef_features(ms_features[i], ms_features[j].detach())
                fd_loss_sum += covar_val
                fd_loss_count += 1
            fd_loss = fd_loss_sum / fd_loss_count
            fd_loss_counter.update(fd_loss.item(), fd_loss_count)
            log_content += ' | loss_fd: {}'.format(format(fd_loss.item(), '.8f'))
        else:
            fd_loss = torch.tensor(0.).to(args.device)
            fd_loss_counter.update(0., 1)
        # endregion

        # region 2.7 pseudo max-correlation loss
        if args.lambda_mc > 0:
            loss_mc_sum, loss_mc_count = torch.tensor(0.).to(args.device), 0
            for stIdx in range(args.stream_num):
                loss_mc_mt_sum, loss_mc_mt_count = torch.tensor(0.).to(args.device), 0
                # region 2.7.1 labeled data
                if args.mc_labeled:
                    corr_val, corr_count = bus.corrcoef_labeled(ms_logits_x[stIdx], targets_x.long().to(args.device), args)
                    if not torch.isnan(corr_val):
                        loss_mc_mt_sum += torch.tensor(1.0).to(args.device) - corr_val
                        loss_mc_mt_count += 1
                # endregion

                # region 2.7.2 unlabeled data
                if args.mc_unlabeled:
                    corr_val, corr_count = bus.corrcoef_unlabeled(ms_logits_u_s[stIdx], targets_ens.detach(), masks, args)
                    if not torch.isnan(corr_val):
                        loss_mc_mt_sum += torch.tensor(1.0).to(args.device) - corr_val
                        loss_mc_mt_count += 1
                # endregion
                mc_loss_counters[stIdx].update(loss_mc_mt_sum.item() / max(1, loss_mc_mt_count), loss_mc_mt_count)
                loss_mc_sum += loss_mc_mt_sum
                loss_mc_count += loss_mc_mt_count
            mc_loss = loss_mc_sum / max(1, loss_mc_count)
            mc_loss_counters[-1].update(mc_loss.item(), loss_mc_count)
            log_content += ' | loss_mc: {}'.format(format(mc_loss.item(), '.8f'))
        else:
            mc_loss = torch.tensor(0.).to(args.device)
            for counter in mc_loss_counters: counter.update(0., 1)
        # endregion

        # region 2.8 calculate total loss & update model
        if args.epo == 0 and batch_idx < args.ensemble_warmup:
            loss = labeled_loss
        else:
            loss = labeled_loss + args.lambda_ens * ensemble_loss + args.lambda_fd * fd_loss + args.lambda_mc * mc_loss
        total_loss_counter.update(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if args.use_ema: model_ema.update(model)
        # endregion

        # region 2.9 visualization
        if args.debug and (args.epo+1) % 5 == 0:
            vis_bat_idxs = [args.eval_step]  # [int(args.eval_step / 2), args.eval_step]
            # 2.9.1 feature visualization
            if (batch_idx+1) in vis_bat_idxs:
                for stIdx in range(args.stream_num):
                    fv_bs = 0
                    fv_val = ms_features[stIdx, fv_bs]
                    save_path = '{}/{}/feature_visualization/epo{}/b{}_bs{}_st{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1, fv_bs+1, stIdx+1)
                    proj.feature_visualize(fv_val, save_path)
                    save_path = '{}/{}/feature_visualization2/epo{}/b{}_bs{}_st{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1, fv_bs+1, stIdx+1)
                    proj.feature_visualize2(fv_val, save_path)
            # endregion

            # 2.9.2 pseudo-labels distribution visualization
            # region pseudo-labels distribution statistic
            stat_bat_idxs = []
            for vis_bat_idx in vis_bat_idxs: stat_bat_idxs += [(idx+1) for idx in range(vis_bat_idx-10, vis_bat_idx)]
            if (batch_idx + 1) in stat_bat_idxs:
                for stIdx in range(args.stream_num + 1):
                    if stIdx == args.stream_num:
                        stat_pl_preds[stIdx] = targets_ens if stat_pl_preds[stIdx] is None else torch.cat([stat_pl_preds[stIdx], targets_ens])
                        stat_pl_masks[stIdx] = masks if stat_pl_masks[stIdx] is None else torch.cat([stat_pl_masks[stIdx], masks])
                    else:
                        stat_pl_preds[stIdx] = ms_max_idx[stIdx] if stat_pl_preds[stIdx] is None else torch.cat([stat_pl_preds[stIdx], ms_max_idx[stIdx]])
                        stat_pl_masks[stIdx] = ms_masks[stIdx] if stat_pl_masks[stIdx] is None else torch.cat([stat_pl_masks[stIdx], ms_masks[stIdx]])
                stat_pl_targets = targets_u if stat_pl_targets is None else torch.cat([stat_pl_targets, targets_u])
            # endregion

            # region pseudo-labels distribution visualization
            if (batch_idx + 1) in vis_bat_idxs:
                # # stream branch predictions
                # for stIdx in range(args.stream_num):
                #     dist_box_preds = bus.target_statistic_train(args, stat_pl_preds[stIdx], stat_pl_targets)
                #     save_path = '{}/{}/distribution_visualization/prediction/e{}_b{}_st{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1, stIdx+1)
                #     proj.distribution_visualize(dist_box_preds, save_path, args.name_classes)
                #
                #     dist_box_preds_sel = bus.target_statistic_train(args, stat_pl_preds[stIdx], stat_pl_targets, stat_pl_masks[stIdx])
                #     save_path = '{}/{}/distribution_visualization/prediction_sel/e{}_b{}_st{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1, stIdx+1)
                #     proj.distribution_visualize(dist_box_preds_sel, save_path, args.name_classes)

                # ensemble prediction
                dist_box_preds_sel = bus.target_statistic_train(args, stat_pl_preds[-1], stat_pl_targets, stat_pl_masks[-1])
                save_path = '{}/{}/distribution_visualization/pseudo_labels/e{}_b{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1)
                proj.distribution_visualize(dist_box_preds_sel, save_path, args.name_classes)

                stat_pl_preds = [None for _ in range(args.stream_num + 1)]
                stat_pl_masks = [None for _ in range(args.stream_num + 1)]
                stat_pl_targets = None
            # endregion
        # endregion
        # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == 1024: print(log_content)

    # region 3. calculate records
    total_loss_val = total_loss_counter.avg
    labeled_losses = [counter.avg for counter in labeled_loss_counters]
    ensemble_losses = [counter.avg for counter in ensemble_loss_counters]
    fd_loss_val = fd_loss_counter.avg
    mc_losses = [counter.avg for counter in mc_loss_counters]
    mask_probs_val = mask_probs_counter.avg
    pl_acc_val = pl_acc_counter.avg
    sim_sum_val = sim_num_counter.sum
    wrong_sum_val = wrong_num_counter.sum
    sim_rate_val = sim_sum_val/wrong_sum_val if wrong_sum_val > 0 else 1.0
    # endregion
    return total_loss_val, labeled_losses, ensemble_losses, fd_loss_val, mc_losses, mask_probs_val, pl_acc_val, sim_sum_val, wrong_sum_val, sim_rate_val


def validate(test_loader, model, args):
    # region 1. Preparation
    test_loss_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    top1_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    top5_counters = [AvgCounter() for _ in range(args.stream_num + 1)]
    predsArray = [[] for _ in range(args.stream_num+1)]

    labelsArray = []
    model.eval()
    # endregion

    # region 2. test
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # region 2.1 data organize
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            labelsArray += targets.clone().cpu().data.numpy().tolist()
            # endregion

            # region 2.2 forward
            ms_logits, _ = model(inputs)
            # endregion

            # region 2.3 calculate loss & accuracy
            prec_list = []
            for stIdx in range(args.stream_num):
                # cal test loss
                logits = ms_logits[stIdx]
                test_loss = F.cross_entropy(logits, targets)
                test_loss_counters[stIdx].update(test_loss.item(), inputs.shape[0])

                # single stream prediction accuracy
                prec1, prec5 = ClassAccuracy.accuracy(logits, targets, topk=(1, 5))
                top1_counters[stIdx].update(prec1.item(), inputs.shape[0])
                top5_counters[stIdx].update(prec5.item(), inputs.shape[0])
                prec_list.append(prec1.item())

                # single stream prediction
                _, preds = torch.max(logits, -1)
                predsArray[stIdx] += preds.clone().cpu().data.numpy().tolist()
            # endregion

            # region 2.4 ensemble prediction
            ms_logits_ens = torch.mean(ms_logits, dim=0)
            _, ms_logits_ens_preds = torch.max(ms_logits_ens, dim=-1)

            frequency = []
            for stIdx in range(args.stream_num):
                logits = ms_logits[stIdx]
                _, logits_preds = torch.max(logits, dim=-1)
                frequency.append(torch.eq(ms_logits_ens_preds, logits_preds).float().sum())

            weighted = torch.softmax(torch.stack(frequency, dim=0), dim=0)
            weighted_ms_logits = []
            for stIdx in range(args.stream_num):
                weighted_ms_logits.append(ms_logits[stIdx]*weighted[stIdx])
            weighted_ms_logits = torch.stack(weighted_ms_logits, dim=0)
            logits_ens = torch.sum(weighted_ms_logits, dim=0)
            ens_loss = F.cross_entropy(logits_ens, targets)
            test_loss_counters[-1].update(ens_loss.item(), inputs.shape[0])

            # get prediction
            _, preds_ens = torch.max(logits_ens, -1)
            predsArray[-1] += preds_ens.clone().cpu().data.numpy().tolist()

            # cal the accuracy
            prec1_ens, prec5_ens = ClassAccuracy.accuracy(logits_ens, targets, topk=(1, 5))
            top1_counters[-1].update(prec1_ens.item(), inputs.shape[0])
            top5_counters[-1].update(prec5_ens.item(), inputs.shape[0])
            # endregion

        # region 3. calculate records
        test_losses = [counter.avg for counter in test_loss_counters]
        top1s = [counter.avg for counter in top1_counters]
        top5s = [counter.avg for counter in top5_counters]
        # endregion

        # region 4 prediction distribution visualization
        if args.debug and (args.epo+1) % 5 == 0 and (batch_idx + 1) == args.eval_step:
            dist_box = bus.target_statistic_infer(args, torch.from_numpy(np.array(predsArray[-1])), torch.from_numpy(np.array(labelsArray)))
            save_path = '{}/{}/distribution_visualization/e{}_b{}.jpg'.format(glob.expr, args.experiment, args.epo+1, batch_idx+1)
            proj.distribution_visualize(dist_box, save_path)
        # endregion
    return predsArray, test_losses, top1s, top5s


def init_args(params=None):
    parser = argparse.ArgumentParser(description='FixMatch Training')

    # Model Setting
    parser.add_argument('--arch', default='WideResNet_MS', type=str, choices=['WideResNet_MS'], help='model name')
    parser.add_argument('--use-ema', default='True')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')

    # Mulitple Stream
    parser.add_argument('--stream-num', default=5, type=int, help='number of stream')
    parser.add_argument('--ensemble-warmup', default=30, type=int, help='number of iteration')
    parser.add_argument('--noisy-factor', default=0.2, type=float)
    parser.add_argument('--blank-num', default=1, type=float)
    parser.add_argument('--lambda-ens', default=1, type=float, help='coefficient of ensemble prediction loss')
    parser.add_argument('--lambda-fd', default=1, type=float, help='coefficient of multi-view features decorrelation loss')
    parser.add_argument('--lambda-mc', default=1, type=float, help='coefficient of max-correlation loss')
    parser.add_argument('--mc-labeled', default='True', help='using labeled data in max-correlation loss')
    parser.add_argument('--mc-unlabeled', default='False', help='using unlabeled data in max-correlation loss')
    parser.add_argument('--count-thr', default=4, type=float, help='threshold of stream count')
    parser.add_argument('--score-thr', default=0.9, type=float, help='threshold of confidence score')

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
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')

    # misc
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--seed', default=1388, type=int, help='random seed')
    parser.add_argument('--debug', default='True', help='do debug operation')

    # params set-up
    args = proj.project_args_setup(parser.parse_args(), params)
    return args


if __name__ == '__main__':
    dataset, num_labeled = "CIFAR10", 40
    # dataset, num_labeled = "CIFAR100", 400
    main('CBE_FixMatch')
