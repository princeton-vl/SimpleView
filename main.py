import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
from dataloader import create_dataloader
from time import time
from datetime import datetime
from progressbar import ProgressBar
import models
from collections import defaultdict
import os
import numpy as np
import argparse
from all_utils import (
    TensorboardManager, PerfTrackTrain,
    PerfTrackVal, TrackTrain, smooth_loss, DATASET_NUM_CLASS,
    rscnn_voting_evaluate_cls, pn2_vote_evaluate_cls)
from configs import get_cfg_defaults
import pprint
from pointnet_pyt.pointnet.model import feature_transform_regularizer
import sys


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print('WARNING: Using CPU')


def check_inp_fmt(task, data_batch, dataset_name):
    if task in ['cls', 'cls_trans']:
        assert set(data_batch.keys()) == {'pc', 'label'}
        pc, label = data_batch['pc'], data_batch['label']
        # special case made for modelnet40_dgcnn to match the
        # original implementation
        # dgcnn loads torch.DoubleTensor for the test dataset
        if dataset_name == 'modelnet40_dgcnn':
            assert isinstance(pc, torch.FloatTensor) or isinstance(
                pc, torch.DoubleTensor)
        else:
            assert isinstance(pc, torch.FloatTensor)
        assert isinstance(label, torch.LongTensor)
        assert len(pc.shape) == 3
        assert len(label.shape) == 1
        b1, _, y = pc.shape[0], pc.shape[1], pc.shape[2]
        b2 = label.shape[0]
        assert b1 == b2
        assert y == 3
        assert label.max().item() < DATASET_NUM_CLASS[dataset_name]
        assert label.min().item() >= 0
    else:
        assert NotImplemented


def check_out_fmt(task, out, dataset_name):
    if task == 'cls':
        assert set(out.keys()) == {'logit'}
        logit = out['logit']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    elif task == 'cls_trans':
        assert set(out.keys()) == {'logit', 'trans_feat'}
        logit = out['logit']
        trans_feat = out['trans_feat']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert isinstance(trans_feat, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert len(trans_feat.shape) == 3
        assert trans_feat.shape[0] == logit.shape[0]
        # 64 coming from pointnet implementation
        assert (trans_feat.shape[1] == trans_feat.shape[2]) and (trans_feat.shape[1] == 64)
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    else:
        assert NotImplemented


def get_inp(task, model, data_batch, batch_proc, dataset_name):
    check_inp_fmt(task, data_batch, dataset_name)
    if not batch_proc is None:
        data_batch = batch_proc(data_batch, DEVICE)
        check_inp_fmt(task, data_batch, dataset_name)

    if isinstance(model, nn.DataParallel):
        model_type = type(model.module)
    else:
        model_type = type(model)

    if task in ['cls', 'cls_trans']:
        pc = data_batch['pc']
        inp = {'pc': pc}
    else:
        assert False

    return  inp


def get_loss(task, loss_name, data_batch, out, dataset_name):
    """
    Returns the tensor loss function
    :param task:
    :param loss_name:
    :param data_batch: batched data; note not applied data_batch
    :param out: output from the model
    :param dataset_name:
    :return: tensor
    """
    check_out_fmt(task, out, dataset_name)
    if task == 'cls':
        label = data_batch['label'].to(out['logit'].device)
        if loss_name == 'cross_entropy':
            loss = F.cross_entropy(out['logit'], label)
        # source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
        elif loss_name == 'smooth':
            loss = smooth_loss(out['logit'], label)
        else:
            assert False
    elif task == 'cls_trans':
        label = data_batch['label'].to(out['logit'].device)
        trans_feat = out['trans_feat']
        logit = out['logit']
        if loss_name == 'cross_entropy':
            loss = F.cross_entropy(logit, label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        elif loss_name == 'smooth':
            loss = smooth_loss(logit, label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        else:
            assert False
    else:
        assert False

    return loss


def validate(task, loader, model, dataset_name):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0

    with torch.no_grad():
        bar = ProgressBar(max_value=len(loader))
        time5  = time()
        for i, data_batch in enumerate(loader):
            time1 = time()
            inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)

            time2 = time()
            out = model(**inp)

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    return perf.agg()


def train(task, loader, model, optimizer, loss_name, dataset_name):
    model.train()

    def get_extra_param():
       return None

    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0

    time3  = time()
    for i, data_batch in enumerate(loader):
        time1 = time()

        inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)
        out = model(**inp)
        loss = get_loss(task, loss_name, data_batch, out, dataset_name)

        perf.update_all(data_batch=data_batch, out=out, loss=loss)
        time2 = time()

        if loss.ne(loss).any():
            print("WARNING: avoiding step as nan in the loss")
        else:
            optimizer.zero_grad()
            loss.backward()
            bad_grad = False
            for x in model.parameters():
                if x.grad is not None:
                    if x.grad.ne(x.grad).any():
                        print("WARNING: nan in a gradient")
                        bad_grad = True
                        break
                    if ((x.grad == float('inf')) | (x.grad == float('-inf'))).any():
                        print("WARNING: inf in a gradient")
                        bad_grad = True
                        break

            if bad_grad:
                print("WARNING: avoiding step as bad gradient")
            else:
                optimizer.step()

        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)

        if i % 50 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")

    return perf.agg(), perf.agg_loss()


def save_checkpoint(id, epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    path = f"./runs/{cfg.EXP.EXP_ID}/model_{id}.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_best_checkpoint(model, cfg):
    path = f"./runs/{cfg.EXP.EXP_ID}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)


def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state'])
            model = model.module

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    # for backward compatibility with saved models
    if 'lr_sched_state' in checkpoint:
        lr_sched.load_state_dict(checkpoint['lr_sched_state'])
        if checkpoint['bnm_sched_state'] is not None:
            bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
    else:
        print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

    return model


def get_model(cfg):
    if cfg.EXP.MODEL_NAME == 'simpleview':
        model = models.MVModel(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.MV)
    elif cfg.EXP.MODEL_NAME == 'rscnn':
        model = models.RSCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.RSCNN)
    elif cfg.EXP.MODEL_NAME == 'pointnet2':
        model = models.PointNet2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.PN2)
    elif cfg.EXP.MODEL_NAME == 'dgcnn':
        model = models.DGCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'pointnet':
        model = models.PointNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    else:
        assert False

    return model


def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    else:
        assert False
    return metric


def get_optimizer(optim_name, tr_arg, model):
    if optim_name == 'vanilla':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=tr_arg.lr_decay_factor,
            patience=tr_arg.lr_reduce_patience,
            verbose=True,
            min_lr=tr_arg.lr_clip)
        bnm_sched = None
    else:
        assert False

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg, resume=False, model_path=""):
    loader_train = create_dataloader(split='train', cfg=cfg)
    loader_valid = create_dataloader(split='valid', cfg=cfg)
    loader_test  = create_dataloader(split='test',  cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)

    if resume:
        model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    else:
        assert model_path == ""


    log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = TensorboardManager(log_dir)
    track_train = TrackTrain(early_stop_patience=cfg.TRAIN.early_stop)

    for epoch in range(cfg.TRAIN.num_epochs):
        print(f'Epoch {epoch}')

        print('Training..')
        train_perf, train_loss = train(cfg.EXP.TASK, loader_train, model, optimizer, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET)
        pprint.pprint(train_perf, width=80)
        tb.update('train', epoch, train_perf)

        if (not cfg.EXP_EXTRA.no_val) and epoch % cfg.EXP_EXTRA.val_eval_freq == 0:
                print('\nValidating..')
                val_perf = validate(cfg.EXP.TASK, loader_valid, model, cfg.EXP.DATASET)
                pprint.pprint(val_perf, width=80)
                tb.update('val', epoch, val_perf)
        else:
            val_perf = defaultdict(float)

        if (not cfg.EXP_EXTRA.no_test) and (epoch % cfg.EXP_EXTRA.test_eval_freq == 0):
            print('\nTesting..')
            test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET)
            pprint.pprint(test_perf, width=80)
            tb.update('test', epoch, test_perf)
        else:
            test_perf = defaultdict(float)

        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf(cfg.EXP.TASK, train_perf, cfg.EXP.METRIC),
            val_metric=get_metric_from_perf(cfg.EXP.TASK, val_perf, cfg.EXP.METRIC),
            test_metric=get_metric_from_perf(cfg.EXP.TASK, test_perf, cfg.EXP.METRIC))

        if (not cfg.EXP_EXTRA.no_val) and track_train.save_model(epoch_id=epoch, split='val'):
            print('Saving best model on the validation set')
            save_checkpoint('best_val', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_test) and track_train.save_model(epoch_id=epoch, split='test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_val) and track_train.early_stop(epoch_id=epoch):
            print(f"Early stopping at {epoch} as val acc did not improve for {cfg.TRAIN.early_stop} epochs.")
            break

        if (not (cfg.EXP_EXTRA.save_ckp == 0)) and (epoch % cfg.EXP_EXTRA.save_ckp == 0):
            save_checkpoint(f'{epoch}', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if cfg.EXP.OPTIMIZER == 'vanilla':
            assert bnm_sched is None
            lr_sched.step(train_loss)
        else:
            assert False

    print('Saving the final model')
    save_checkpoint('final', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

    print('\nTesting on the final model..')
    last_test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET)
    pprint.pprint(last_test_perf, width=80)

    tb.close()


def entry_test(cfg, test_or_valid, model_path=""):
    split = "test" if test_or_valid else "valid"
    loader_test = create_dataloader(split=split, cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)
    model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    model.eval()

    test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET)
    pprint.pprint(test_perf, width=80)
    return test_perf


def rscnn_vote_evaluation(cfg, model_path, log_file):
    model = get_model(cfg)
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        print("WARNING: using dataparallel to load data")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state'])
    print(f"Checkpoint loaded from {model_path}")
    model.to(DEVICE)
    model.eval()

    assert cfg.EXP.DATASET in ["modelnet40_rscnn"]
    loader_test = create_dataloader(split='test', cfg=cfg)

    rscnn_voting_evaluate_cls(
        loader=loader_test,
        model=model,
        data_batch_to_points_target=lambda x: (x['pc'], x['label']),
        points_to_inp=lambda x: {'pc': x},
        out_to_prob=lambda x: F.softmax(x['logit'], dim=1),
        log_file=log_file
    )


def pn2_vote_evaluation(cfg, model_path, log_file):
    assert cfg.EXP.DATASET in ["modelnet40_pn2"]
    loader_test = create_dataloader(split='test', cfg=cfg)

    model = get_model(cfg)
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        print("WARNING: using dataparallel to load data")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state'])
    print(f"Checkpoint loaded from {model_path}")
    model.to(DEVICE)
    model.eval()

    pn2_vote_evaluate_cls(loader_test, model, log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument('--exp-config', type=str, default="")
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--resume', action="store_true", default=False)

    cmd_args = parser.parse_args()

    if cmd_args.entry == "train":
        assert not cmd_args.exp_config == ""
        if not cmd_args.resume:
            assert cmd_args.model_path == ""

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cmd_args.exp_config)
        if cfg.EXP.EXP_ID == "":
            cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
        cfg.freeze()
        print(cfg)

        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        entry_train(cfg, cmd_args.resume, cmd_args.model_path)

    elif cmd_args.entry in ["test", "valid"]:
        assert not cmd_args.exp_config == ""
        assert not cmd_args.model_path == ""

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cmd_args.exp_config)
        cfg.freeze()
        print(cfg)

        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        test_or_valid = cmd_args.entry == "test"
        entry_test(cfg, test_or_valid, cmd_args.model_path)

    elif cmd_args.entry in ["rscnn_vote", "pn2_vote"]:
        assert not cmd_args.exp_config == ""
        assert not cmd_args.model_path == ""
        log_file = f"vote_log/{cmd_args.model_path.replace('/', '_')}_{cmd_args.entry.replace('/', '_')}.log"

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cmd_args.exp_config)
        cfg.freeze()
        print(cfg)

        seed  = cfg.EXP.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        if cmd_args.entry == "rscnn_vote":
            rscnn_vote_evaluation(cfg, cmd_args.model_path, log_file)
        elif cmd_args.entry == "pn2_vote":
            pn2_vote_evaluation(cfg, cmd_args.model_path, log_file)
    else:
        assert False
