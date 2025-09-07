1. python3 run_rl_task.py --task_yaml examples/multirobot/tasks/multirobot.yaml --iterations 400 --use_default_curriculum --batch_mode complete_episodes --rollout_fragment_length 256 --train_batch_size 6144 --sgd_minibatch_size 128 --lr 1e-4

2. python3 run_rl_task.py --task_yaml examples/multirobot/tasks/multirobot.yaml --iterations 100 --use_default_curriculum --batch_mode complete_episodes --rollout_fragment_length 256 --train_batch_size 8192 --sgd_minibatch_size 128 --lr 1e-4 --headless --checkpoint_path checkpoint

3. python3 log_analyzer.py --csv prevlogs/training_metrics.csv --tail-n 100 --clean-outdir --curriculum-boundaries 45,75,135 --jsonl-pattern prevlogs/training_metrics.jsonl

4. python3 src/scripts/export_onnx.py --checkpoint_dir checkpoint --task_yaml examples/multirobot/tasks/multirobot.yaml --out_dir onnx_export --policy_name shared_policy