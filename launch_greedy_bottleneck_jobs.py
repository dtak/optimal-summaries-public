import os
import re
import time
import argparse
import random
import csv

gpu_slurm_template = """#!/bin/bash
#SBATCH --mem=10000
#SBATCH -t 0-60:00
#SBATCH -p gpu
#SBATCH -o {out_file}
#SBATCH -e {err_file}
#SBATCH --gres=gpu:1
#SBATCH -n 1
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01 Anaconda3/5.0.1-fasrc01
source activate nj
{job_command}
"""

cpu_slurm_template = """#!/bin/bash
#SBATCH --mem=10000
#SBATCH -t 0-3:00
#SBATCH -p doshi-velez
#SBATCH -o {out_file}
#SBATCH -e {err_file}
#SBATCH -n 4
module load Anaconda3/5.0.1-fasrc01
source activate nj
{job_command}
"""

parser = argparse.ArgumentParser()
parser.add_argument('--hardware', type=str, default='gpu')
parser.add_argument('--timestamp', type=str, default='')
FLAGS = parser.parse_args()

if FLAGS.hardware == 'gpu':
    slurm_template = gpu_slurm_template
elif FLAGS.hardware == 'cpu':
    slurm_template = cpu_slurm_template
else:
    raise ValueError('unknown hardware')

if len(FLAGS.timestamp) > 0:
    timestamp = FLAGS.timestamp
else:
    timestamp = int(time.time())

save_dir = "insert_dir_name"
os.system(f"mkdir -p {save_dir}")

def launch_job(exp, time_limit=None, mem_limit=None):

  job_command = "python3 -u gpu_greedy_top_concepts.py"

  for k, v in exp.items():
    job_command += f" --{k}={v}"

  out_file = os.path.join(save_dir, 'job-%j.out')
  err_file = os.path.join(save_dir, 'job-%j.err')
  slurm_file = os.path.join(save_dir, 'job.slurm')

  slurm_command = slurm_template.format(
    job_command=job_command,
    out_file=out_file,
    err_file=err_file)
  with open(slurm_file, "w") as f: f.write(slurm_command)

  os.system("cat {} | sbatch".format(slurm_file))

for r in range(1,4):
    d = {}
    d['split_random_state'] = r
    launch_job(d)

