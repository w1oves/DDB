import os
import os.path as osp
import unittest
from mmcv import Config
from dass.models import build_uda_segmentor


class TestModels(unittest.TestCase):
    def build_config(self, config_file):
        config = Config.fromfile(config_file)
        build_uda_segmentor(config)
        build_uda_segmentor(config)

    def test_models_config(self):
        config_dir = "configs"
        for models_config_dir in os.listdir(config_dir):
            if models_config_dir == "_base_":
                continue
            models_config_dir_path = osp.join(config_dir, models_config_dir)
            for config in os.listdir(models_config_dir_path):
                config_path = osp.join(models_config_dir_path, config)
                self.build_config(config_path)
                print(f"test for {config} passed")
