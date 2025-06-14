#!/bin/bash

python3 mpc.py --eval_log True --save_path "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/classical_methods/run2/mpc/"
sleep 60

python3 pure_pursuit.py --eval_log True --save_path "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/classical_methods/run2/pure_pursuit/"
sleep 60

python3 PDCenter.py --eval_log True --save_path "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/classical_methods/run2/pd_cent/"
sleep 60

python3 RLPolicy.py --policy ImgCent --eval_log True --save_path "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/classical_methods/run2/rl_img_cent/"
sleep 60

python3 RLPolicy.py --policy bslnPEVN --eval_log True --save_path "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/classical_methods/run2/rl_pevn/"
sleep 60