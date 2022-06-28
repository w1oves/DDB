import os
import unittest
from mmcv import Config
from dass.datasets import build_uda_dataset


class TestDataset(unittest.TestCase):
    def build_config(self, config_file):
        config = Config.fromfile(config_file)
        build_uda_dataset(config.data.train)
        build_uda_dataset(config.data.val)

    def test_datasets_config(self):
        datasets_config_dir = "configs/_base_/datasets"
        for config in os.listdir(datasets_config_dir):
            if config.startswith("uda") or config.startswith('test'):
                continue
            config_path = os.path.join(datasets_config_dir, config)
            self.build_config(config_path)
            print(f"test for {config} passed")
