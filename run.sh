python steerability.py --pres present.txt --past past.txt --walk_file walk_rand_init.pt --init_mode rand > results_05_11_21/walk_rand_init.log

python steerability.py --pres present.txt --past past.txt --walk_file walk_zero_init.pt --init_mode zero > results_05_11_21/walk_zero_init.log

python steerability.py --pres present.txt --past past.txt --walk_file walk_arith_init.pt --init_mode arithmetic > results_05_11_21/walk_arith_init.log

python steerability.py --pres repeat_present.txt --past repeat_past.txt --walk_file walk_repeat_rand_init.pt --init_mode rand > results_05_11_21/walk_repeat_rand_init.log

python steerability.py --pres repeat_present.txt --past repeat_past.txt --walk_file walk_repeat_zero_init.pt --init_mode rand > results_05_11_21/walk_repeat_zero_init.log

python steerability.py --pres combined_present.txt --past combined_past.txt --walk_file walk_combined_rand_init.pt --init_mode rand > results_05_11_21/walk_combined_rand_init.log


