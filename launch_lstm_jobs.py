import os
import re
import time
import argparse
import random
import csv


gpu_slurm_template = """#!/bin/bash
#SBATCH --mem=10000
#SBATCH -t 0-8:00
#SBATCH -p gpu_requeue
#SBATCH -o {out_file}
#SBATCH -e {err_file}
#SBATCH --gres=gpu:1
#SBATCH -n 4
module load git python/3.6.3-fasrc01 cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
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

  job_command = "python3 -u gpu_lstm_baseline_compound.py"

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
split_random_state = 1

arr_lstm_hidden_dim = [4, 8, 16, 32, 64, 128, 256]
arr_hidden_next_state_dim = [32, 64, 128, 256, 512, 1024, 2048]
arr_hidden_outcome_dim = [32, 64, 128, 256, 512, 1024, 2048]
arr_alpha = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
arr_beta = [0.1, 1, 10, 20, 40, 60, 100]
arr_opt_lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
arr_opt_weight_decay = [0, 1e-5, 1e-4, 1e-3]

num_epochs = 10000
save_every = 250

N_experiments = 150

dir_path = '/n/home07/carissawu/optimal-summaries/vasopressor/models/LOS-6-600/lstm/'
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

rs = [10]
for r in rs:
    # sbatch N_experiments experiments with randomly sampled
    random.seed(1)
    filename = "vasopressor_lstm_gridsearch_r{}".format(r)
    fields = ['lstm_hidden_dim', 'hidden_next_state_dim', 'hidden_outcome_dim', 'alpha', 'beta','opt_lr','opt_weight_decay','test auc']    
    # sbatch N_experiments experiments with randomly sampled hyperparameters
    with open('{file_path}.csv'.format(file_path=os.path.join(dir_path, filename)), 'w+') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)        # Create exp dictionary d by randomly sampling from each of the arrays.
    for n in range(N_experiments):
        d = {}

        d['split_random_state'] = r
        d['num_epochs'] = num_epochs
        d['save_every'] = save_every

        d['lstm_hidden_dim'] = random.choice(arr_lstm_hidden_dim)
        d['hidden_next_state_dim'] = random.choice(arr_hidden_next_state_dim)
        d['hidden_outcome_dim'] = random.choice(arr_hidden_outcome_dim)
        d['alpha'] = random.choice(arr_alpha)
        d['beta'] = random.choice(arr_beta)
        d['opt_lr'] = random.choice(arr_opt_lr)
        d['opt_weight_decay'] = random.choice(arr_opt_weight_decay)

        d['output_dir'] = 'models'
        d['model_output_name'] = 'models/LOS-6-600/lstm/lstm_r' + str(d['split_random_state']) + '_hd_' + str(d['lstm_hidden_dim']) + '_hnsd_' + str(d['hidden_next_state_dim']) + '_hod_' + str(d['hidden_outcome_dim']) + '_alpha_' + str(d['alpha']) + '_beta_' + str(d['beta']) + '_optlr_' + str(d['opt_lr']) + '_optwd_' + str(d['opt_weight_decay']) + '.pt'

        launch_job(d)

