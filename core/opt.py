"""
Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/tools/program.py
"""
from typing import Optional
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml
import json
from core.utils.loading import load_yaml


class Config(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)
        super(Config, self).update(config)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        print(f"Saving config to {path}...")
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path):
        print(f"Loading config from {path}...")
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


class Opts(ArgumentParser):

    def __init__(self, cfg: Optional[str] = None):
        super(Opts, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c",
                          "--config",
                          default=cfg,
                          help="configuration file to use")
        self.add_argument("-o",
                          "--opt",
                          nargs="+",
                          help="override configuration options")

    def parse_args(self, argv=None):
        args = super(Opts, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)

        config = Config(args.config)
        config = self.override(config, args.opt)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        print("Overriding configurating")
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    print(f"'{key}' not found in config")
            else:
                sub_keys = key.split(".")
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0])
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            print(f"'{key}' not found in config")
                    else:
                        cur = cur[sub_key]
        return global_config

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from typing import Any, Callable, Dict, IO, List, Union

import yaml
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as _CfgNode


BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:

    1. The :meth:`merge_from_file` method supports the "_BASE_" key,
       which allows the new CfgNode to inherit all the attributes from the
       base configuration file(s).
    2. Keys that start with "COMPUTED_" are treated as insertion-only
       "computed" attributes. They can be inserted regardless of whether
       the CfgNode is frozen or not.
    3. With "allow_unsafe=True", it supports pyyaml tags that evaluate
       expressions in config. See examples in
       https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    """

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        """
        Defines how a config file is opened. May be overridden to support
        different file schemas.
        """
        return g_pathmgr.open(filename, "r")

    @classmethod
    def load_yaml_with_base(
        cls, filename: str, allow_unsafe: bool = False
    ) -> Dict[str, Any]:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)

        def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        def _load_with_base(base_cfg_file: str) -> Dict[str, Any]:
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            return cls.load_yaml_with_base(base_cfg_file, allow_unsafe=allow_unsafe)

        if BASE_KEY in cfg:
            if isinstance(cfg[BASE_KEY], list):
                base_cfg: Dict[str, Any] = {}
                base_cfg_files = cfg[BASE_KEY]
                for base_cfg_file in base_cfg_files:
                    merge_a_into_b(_load_with_base(base_cfg_file), base_cfg)
            else:
                base_cfg_file = cfg[BASE_KEY]
                base_cfg = _load_with_base(base_cfg_file)
            del cfg[BASE_KEY]

            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False) -> None:
        """
        Merge configs from a given yaml file.

        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    # Forward the following calls to base, but with a check on the BASE_KEY.
    def merge_from_other_cfg(self, cfg_other: "CfgNode") -> Callable[[], None]:
        """
        Args:
            cfg_other (CfgNode): configs to merge from.
        """
        assert (
            BASE_KEY not in cfg_other
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list: List[str]) -> Callable[[], None]:
        """
        Args:
            cfg_list (list): list of configs to merge from.
        """
        keys = set(cfg_list[0::2])
        assert (
            BASE_KEY not in keys
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_list(cfg_list)

    def __setattr__(self, name: str, val: Any) -> None:  # pyre-ignore
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                raise KeyError(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(name, old_val, val)
                )
            self[name] = val
        else:
            super().__setattr__(name, val)