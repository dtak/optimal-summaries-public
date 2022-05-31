import os
import re
import time
import argparse
import random
import csv


gpu_slurm_template = """#!/bin/bash
#SBATCH --mem=10000
#SBATCH -t 0-100:00
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

  job_command = "python3 -u gpu_vasopressor_bottleneck_finetune.py"

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

# Run experiments with randomly sampled hyperparameters.
arr_opt_lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
arr_opt_weight_decay = [0, 1e-5, 1e-4, 1e-3]

num_epochs = 1000
save_every = 10

dir_path = '/n/home07/carissawu/optimal-summaries/vasopressor/models/LOS-6-600/cos-sim/top-k'
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

for r in range(1,11):
    for c in range(1,5):
        # Write hyperparameters to csv file
        fields = ['num_concepts', 'opt_lr', 'opt_weight_decay', 'l1_lambda', 'cos_sim_lambda','test auc']
        with open('{file_path}.csv'.format(file_path=os.path.join(dir_path, filename)), 'w+') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(fields)
        random.seed(1)
        for n in range(N_experiments):
            # Create exp dictionary d by randomly sampling from each of the arrays.
            for lr in arr_opt_lr:
                for wd in arr_opt_weight_decay:
                    d = {}

                    d['num_concepts'] = c
                    d['split_random_state'] = r
                    d['num_epochs'] = num_epochs
                    d['save_every'] = save_every

                    d['opt_lr'] = lr
                    d['opt_weight_decay'] = wd

                    d['output_dir'] = 'models/LOS-6-600/no-reg'
                    d['model_output_name'] = 'models/LOS-6-600/no-reg/bottleneck_r' + str(r) + '_c' + str(d['num_concepts']) + '_optlr_' + str(d['opt_lr']) + '_optwd_' + str(d['opt_weight_decay']) + '.pt'

                    launch_job(d)

