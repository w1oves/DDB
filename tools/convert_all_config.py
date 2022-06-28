# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from mmcv import Config

FLAG = "____pretty_text____"


def convert(path: str) -> bool:
    try:
        text = Config.fromfile(path).pretty_text
    except:
        print(f"convert {path} False")
        text = "False to Parse"
    flag_inherit = False
    with open(path, "r", encoding="utf-8") as f:
        content = []
        for line in f:
            if line.find(FLAG) != -1:
                break
            if not flag_inherit and line.find("_base_") != -1:
                flag_inherit = True
            content.append(line)
    if flag_inherit:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(content)
            f.writelines(["\n"])
            f.write(f"'''###{FLAG}###'''")
            f.writelines(["\n"] * 4)
            f.write("'''\n")
            f.write(text)
            f.write("'''\n")
        print(f"convert {path} True")
    return True


from multiprocessing import Pool
import os, time, random


def main():
    start = time.time()
    config_dir = "configs"
    configs = []
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith(".py"):
                configs.append(osp.join(root, file))
    num_configs = len(configs)
    p = Pool(32)
    for config in configs:
        p.apply_async(convert, args=(config,))
    p.close()
    p.join()
    end = time.time()
    print(f"convert {num_configs} files, use {end-start} s")


if __name__ == "__main__":
    main()
