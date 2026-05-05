# -*- coding: utf-8 -*-
import logging
import subprocess
import time
from itertools import product

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # 配置日志记录器
    l = ['cyberbullying']
    batch_sizes = {16}
    learning_rates = {'4e-5'}
    shots = {15}
    seeds = {143}
    template_id = {1}

    verbalizer = {'kpt'}
    for n, t, j, i, k, m, v in product(l,template_id, seeds, batch_sizes, learning_rates, shots, verbalizer):
        cmd = (
            f"python fewshot.py --result_file ./reproduct-cb_result_autoformer.txt "
            f"--dataset {n} --template_id {t} --seed {j} "
            f"--batch_size {i} --shot {m} --verbalizer {v}"
        )

        logging.info(f"Executing command: {cmd}")
        print(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")

        time.sleep(2)
