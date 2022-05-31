import os
import re
import time
import argparse
import random
import csv

gpu_slurm_template = """#!/bin/bash
#SBATCH --mem=10000
#SBATCH -t 0-8:00
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
#SBATCH -t 0-8:00
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

  job_command = "python3 -u gpu_vasopressor_baseline.py"

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
arr_alpha = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
arr_tau = [0.1, 1, 10, 20, 40, 60, 100]
arr_opt_lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
arr_opt_weight_decay = [0, 1e-5, 1e-4, 1e-3]

# output_dir
# model_output_name

num_epochs = 1000
save_every = 10

N_experiments = 30

dir_path = '/n/home07/carissawu/optimal-summaries/vasopressor/models/LOS-6-600/baseline/'
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

for r in range(9,11):
    filename = "vasopressor_baseline_gridsearch_r{}".format(r)
    fields = ['alpha', 'tau', 'opt_lr', 'opt_weight_decay', 'test auc']    
    # sbatch N_experiments experiments with randomly sampled hyperparameters
    with open('{file_path}.csv'.format(file_path=os.path.join(dir_path, filename)), 'w+') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)
    random.seed(1)
    for n in range(N_experiments):
        # Create exp dictionary d by randomly sampling from each of the arrays.
        d = {}

        d['split_random_state'] = r
        d['num_epochs'] = num_epochs
        d['save_every'] = save_every

        d['alpha'] = random.choice(arr_alpha)
        d['tau'] = random.choice(arr_tau)
        d['opt_lr'] = random.choice(arr_opt_lr)
        d['opt_weight_decay'] = random.choice(arr_opt_weight_decay)

        d['output_dir'] = 'models/LOS-6-600/baseline'
        d['model_output_name'] = 'models/LOS-6-600/baseline/baseline_r'+str(r) +'_alpha_' + str(d['alpha']) + '_tau_' + str(d['tau']) + '_optlr_' + str(d['opt_lr']) + '_optwd_' + str(d['opt_weight_decay']) + '.pt'

        launch_job(d)

