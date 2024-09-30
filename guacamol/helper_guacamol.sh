#!/bin/bash
source /root/.bashrc
conda activate retmol_clone

python run_retrieval_ga.py --benchmark_id 7 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'
python run_retrieval_ga.py --benchmark_id 8 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'
python run_retrieval_ga.py --benchmark_id 9 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'
python run_retrieval_ga.py --benchmark_id 10 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'
python run_retrieval_ga.py --benchmark_id 11 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'
python run_retrieval_ga.py --benchmark_id 12 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'
python run_retrieval_ga.py --benchmark_id 13 --n_retrievals 10 --n_repeat 100 --n_trials 100 --batch_size 5 --gen_model 'zinc-pretrain'