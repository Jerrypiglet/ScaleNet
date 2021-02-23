# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
from termcolor import colored

from .model_serialization import load_state_dict
from maskrcnn_rui.utils.c2_model_loading import load_c2_format
from maskrcnn_rui.utils.imports import import_file
from maskrcnn_rui.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        opt,
        model,
        optimizer=None,
        scheduler=None,
        checkpoint_all_dir='',
        save_dir="",
        save_to_disk=None,
        logger=None,
        if_print=True,
        if_reset_scheduler=False,
    ):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_all_dir = checkpoint_all_dir
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.if_print = if_print
        self.if_reset_scheduler = if_reset_scheduler

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        else:
            print(colored("Optimizer not found! Thus not saved!", 'yellow', 'on_red'))
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        else:
            print(colored("scheduler not found! Thus not saved!", 'yellow', 'on_red'))
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info(colored("Saving checkpoint to {}".format(save_file), 'white', 'on_magenta'))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

        return save_file

    def load(self, f=None, use_latest=True, skip_kws=[], only_load_kws=[], replace_kws=[], replace_with_kws=[], task_name=None):
        if f is None:
            if task_name is not None:
                # look for the latest checkpoint in a previous task
                f = self.get_checkpoint_file(task_name)
                self.logger.info("Using 'latest checkpoint' from task %s...; at %s"%(task_name, f))
            else:
                if use_latest and self.has_checkpoint():
                    # override argument with existing checkpoint
                    f = self.get_checkpoint_file()
                    self.logger.info("Using existing 'latest checkpoint'...")
        if not f:
            # no checkpoint could be found
            self.logger.error("No checkpoint found. Initializing model from scratch")
            raise ValueError('No checkpoint found!')
            return {}
        self.logger.info(colored("Loading checkpoint from %s."%f, 'white', 'on_magenta'))
        checkpoint = self._load_file(f)

        current_keys, loaded_keys = self._load_model(checkpoint, self.logger, skip_kws=skip_kws, only_load_kws=only_load_kws, replace_kws=replace_kws, replace_with_kws=replace_with_kws)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            groups = self.optimizer.param_groups
            saved_groups = checkpoint["optimizer"]['param_groups']
            param_lens = (len(g['params']) for g in groups)
            saved_lens = (len(g['params']) for g in saved_groups)
            # for p_len, s_len in zip(param_lens, saved_lens):
            #     print(p_len, s_len)
            # # print(groups[0].keys())
            # print([g.keys() for g in groups])
            if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
                self.logger.info(colored("loaded state dict contains a parameter group that doesn't match the size of optimizer's group! Thus not Restored!", 'yellow', 'on_red'))
            else:
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        else:
            self.logger.info(colored("Optimizer not found! Thus not Restored!", 'yellow', 'on_red'))
            print(checkpoint.keys(), self.optimizer)
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
            if self.if_reset_scheduler:
                self.scheduler._reset()
                self.logger.info("scheduler._reset()")

        else:
            self.logger.info(colored("Scheduler not found! Thus not Restored!", 'yellow', 'on_red'))

        # return any further checkpoint data
        return checkpoint, current_keys, loaded_keys

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self, task_name=None):
        if task_name is not None:
            task_name_splits = task_name.split('/')
            if len(task_name_splits) == 1:
                ckpt_folder = os.path.join(self.checkpoint_all_dir, task_name)
            else:
                ckpt_folder = os.path.join(self.opt.pwdpath, task_name)
        else:
            ckpt_folder = self.save_dir
        save_file = os.path.join(ckpt_folder, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
            self.logger.warning(save_file + 'NOT FOUND!')
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, logger=None, skip_kws=[], only_load_kws=[], replace_kws=[], replace_with_kws=[]):
        current_keys, loaded_keys = load_state_dict(self.model, checkpoint.pop("model"), logger=logger, skip_kws=skip_kws, only_load_kws=only_load_kws, replace_kws=replace_kws, replace_with_kws=replace_with_kws)
        return current_keys, loaded_keys


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        opt,
        model,
        optimizer=None,
        scheduler=None,
        checkpoint_all_dir='',
        save_dir="",
        save_to_disk=None,
        logger=None,
        if_print=True,
        if_reset_scheduler=False
    ):
        super().__init__(
            opt, model, optimizer, scheduler, checkpoint_all_dir, save_dir, save_to_disk, logger, if_print, if_reset_scheduler=if_reset_scheduler
        )
        self.cfg = opt.cfg.clone()
        self.logger = logger

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
