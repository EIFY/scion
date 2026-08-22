#!/bin/bash

scripts=("rerun_power.sh" "rerun_cos_power.sh" "rerun_cosine_power_comparison.sh" "rel_lr.sh" "rerun_full.sh" "rerun_mosch.sh" "rerun_mosch_full.sh" "rerun_log_time_mosch_full.sh" "row_norm_lr.sh" "row_norm_wd.sh" "row_norm_sign_lr.sh" "row_norm_sign_wd.sh" "row_norm_c_sq_lr.sh" "row_norm_lr_eff_transfer.sh" "row_norm_done")

len=${#scripts[@]}

python row_norm_script_gen.py

while [ ! -f row_norm_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python row_norm_script_gen.py
			break
		fi
	done
done

