# FAIML Reinforcement Learning Project

Course project for FAIML - 01VSDWS.

## Getting Started

For reproducible Hopper-v4 results, use Python 3.10 and install the pinned
versions in `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate faiml-rl
```

Then install the local PandaPush package used by Part 2:

```bash
cd part2/panda-gym
pip install -e .
```

The submitted Part 1 run used `mujoco==3.8.1` and `gymnasium==1.2.3`.
MuJoCo physics changes across versions can change Hopper-v4 learning curves
even when the seed is fixed, so rerun Part 1 from a clean environment if these
versions differ. New `part1/train.py` runs also write the runtime versions into
`part1_summary.csv` and the full command/config into
`part1_run_metadata.json`.

Check the simulator stack before comparing results:

```bash
python -c "import gymnasium, mujoco; print(gymnasium.__version__, mujoco.__version__)"
```

For closest reruns, start from a fresh env and compare runs only when the
metadata files show the same Python, Gymnasium, MuJoCo, NumPy, and Torch
versions. RL training can still vary slightly across hardware/BLAS backends, so
the saved results should be treated as reproducible under the pinned stack, not
portable across arbitrary simulator versions.

## Part 1: Hopper-v4

Required from-scratch experiments only:

```bash
cd part1
python test_random_policy.py
python train.py
python evaluate_required_actor_critic_hopper.py
python render_required_actor_critic_hopper.py
```

The `python train.py` command runs the report-matching vanilla REINFORCE and
Actor-Critic experiments and writes the submitted Part 1 model, CSV files, and
learning-curve figure.

Main outputs:

- `part1/part1_summary.csv`
- `part1/part1_comparison_learning_curve.png`
- `part1/best_actor_critic_policy.pth`

## Part 2: PandaPush-v3

Required PPO/SAC training with fixed cube mass:

```bash
cd part2

# PPO
# added --learning-rate 0.0001 for PPO to match the existing models & report
# lower bound PPO 200k
python train_sb3.py --algo ppo --env-type source --sampling-strategy none --timesteps 200000 --learning-rate 0.0001
# upper bound PPO 200k
python train_sb3.py --algo ppo --env-type target --sampling-strategy none --timesteps 200000 --learning-rate 0.0001
# upper bound PPO 500k - I added this for fair comparison
python train_sb3.py --algo ppo --env-type target --sampling-strategy none --timesteps 300000 --load-model-path models/ppo_push_none_target_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --run-name ppo_push_none_target_500k_lr0p0001_g0p95_n1024_b64_fixed_seed42 --learning-rate 0.0001 --quiet

# SAC
# lower bound SAC
python train_sb3.py --algo sac --env-type source --sampling-strategy none --timesteps 200000
# upper bound SAC 200k
python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 200000
# upper bound SAC 500k - I made a slight change this so it saves computing time
python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 300000 --load-model-path models/sac_push_none_target_200k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --run-name sac_push_none_target_500k_lr0p0003_g0p95_buf100000_b256_fixed_seed42 --quiet

# I commented below codes as it's clashing, error or no longer needed
# python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 500000 --quiet
#python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 120000 --load-model-path models/sac_push_none_target_260k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --run-name sac_push_none_target_380k_lr0p0003_g0p95_buf100000_b256_fixed_seed42 --quiet

# I listed all evaluate command below, in details so it's only evaluating specific models we use in the report
# python eval_sb3.py --model-path models/MODEL_NAME.zip --env-type target --episodes 50 --save-csv
```

Evaluate lower/upper bound models (source→source, source→target, target→target):

```bash
cd part2
python eval_sb3.py --model-path models/ppo_push_none_source_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --env-type source --episodes 50 --save-csv --experiment-name ppo_none_source_to_source_50ep
python eval_sb3.py --model-path models/ppo_push_none_source_200k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name ppo_none_source_to_target_50ep
python eval_sb3.py --model-path models/ppo_push_none_target_500k_lr0p0001_g0p95_n1024_b64_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name ppo_none_target_to_target_50ep
python eval_sb3.py --model-path models/sac_push_none_source_200k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --env-type source --episodes 50 --save-csv --experiment-name sac_none_source_to_source_50ep
python eval_sb3.py --model-path models/sac_push_none_source_200k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name sac_none_source_to_target_50ep
python eval_sb3.py --model-path models/sac_push_none_target_500k_lr0p0003_g0p95_buf100000_b256_fixed_seed42.zip --env-type target --episodes 50 --save-csv --experiment-name sac_none_target_to_target_50ep
```

The submitted domain-randomization experiments use the required PPO and SAC
implementations with UDR and ADR:

```bash
cd part2
# PPO
python train_sb3.py --algo ppo --env-type source --sampling-strategy udr --timesteps 200000 --learning-rate 0.0001 --mass-max 5.0 --seed 0
python train_sb3.py --algo ppo --env-type source --sampling-strategy adr --timesteps 200000 --learning-rate 0.0001 --adr-max-limit 5.0 --adr-step 0.3 --adr-success-threshold 0.25 --adr-window-size 10 --seed 0

# SAC
# 200k
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --timesteps 200000 --mass-min 0.5 --mass-max 5.0 --seed 0 --quiet
python train_sb3.py --algo sac --env-type source --sampling-strategy adr --timesteps 200000 --mass-min 0.5 --adr-max-limit 5.0 --adr-initial-min 0.8 --adr-initial-max 1.2 --adr-step 0.1 --adr-success-threshold 0.65 --adr-window-size 30 --seed 0
# 500k
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --timesteps 500000 --mass-min 0.5 --mass-max 5.0 --seed 0
python train_sb3.py --algo sac --env-type source --sampling-strategy adr --timesteps 500000 --mass-min 0.5 --adr-max-limit 5.0 --adr-initial-min 0.8 --adr-initial-max 1.2 --adr-step 0.1 --adr-success-threshold 0.65 --adr-window-size 30 --seed 0
# python train_sb3.py --algo sac --env-type source --sampling-strategy adr --timesteps 200000 --mass-min 0.5 --adr-max-limit 5.0 --adr-initial-min 0.8 --adr-initial-max 1.2 --adr-step 0.3 --adr-success-threshold 0.25 --adr-window-size 10 --run-name sac_push_adr_source_200k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p3_thr0p25_win10_seed0 --quiet
```

Evaluate DR models on target domain + mass robustness sweep:

```bash
cd part2

# PPO DR target evals
python eval_sb3.py --model-path models/ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p5-5p0_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name udr_v3_target
python eval_sb3.py --model-path models/ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p8-3p0_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name udr_v2_target
python eval_sb3.py --model-path models/ppo_push_adr_source_200k_lr0p0001_g0p95_n1024_b64_init0p8-1p2_lim0p5-5p0_step0p3_thr0p25_win10_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name ppo_adr_target

# SAC DR target evals
python eval_sb3.py --model-path models/sac_push_udr_source_200k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name sac_udr_target
python eval_sb3.py --model-path models/sac_push_adr_source_200k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p1_thr0p65_win30_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name sac_adr_200k_conservative_target
python eval_sb3.py --model-path models/sac_push_udr_source_500k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name sac_udr_500k_target
python eval_sb3.py --model-path models/sac_push_adr_source_500k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p1_thr0p65_win30_seed0.zip --env-type target --episodes 50 --save-csv --experiment-name sac_adr_500k_target

# Mass robustness evals
for MASS in 1.0 2.0 3.0 4.0 5.0; do
    python eval_sb3.py --model-path models/ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p5-5p0_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name udr_v3_mass_$MASS
    python eval_sb3.py --model-path models/ppo_push_udr_source_200k_lr0p0001_g0p95_n1024_b64_m0p8-3p0_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name udr_v2_mass_$MASS
    python eval_sb3.py --model-path models/ppo_push_adr_source_200k_lr0p0001_g0p95_n1024_b64_init0p8-1p2_lim0p5-5p0_step0p3_thr0p25_win10_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name ppo_adr_mass_$MASS
    python eval_sb3.py --model-path models/sac_push_udr_source_200k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name sac_udr_mass_$MASS
    python eval_sb3.py --model-path models/sac_push_adr_source_200k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p1_thr0p65_win30_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name sac_adr_200k_conservative_mass_$MASS
    python eval_sb3.py --model-path models/sac_push_udr_source_500k_lr0p0003_g0p95_buf100000_b256_m0p5-5p0_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name sac_udr_500k_mass_$MASS
    python eval_sb3.py --model-path models/sac_push_adr_source_500k_lr0p0003_g0p95_buf100000_b256_init0p8-1p2_lim0p5-5p0_step0p1_thr0p65_win30_seed0.zip --env-type source --eval-mass $MASS --episodes 50 --save-csv --experiment-name sac_adr_500k_mass_$MASS
done

python plot_results.py
```

Optional: render start+finish images for any trained model:

```bash
cd part2
python render_panda.py --model-path models/MODEL_NAME.zip --env-type target --seed 53 --out-dir ../renders
```
Produces two images per model: default angle and zoomed top-down angle, each with a title showing the full training configuration.

Main outputs:

- `part2/results/eval_results.csv`
- `part2/results/domain_randomization_target_barplot.png`
- `part2/results/mean_return_barplot.png`
- `part2/results/mass_robustness_curve.png`
- `part2/results/lower_upper_success_barplot.png`
- `renders/part2_*_start_finish.png` (default angle)
- `renders/part2_*_start_finish_topdown.png` (top-down angle)

New Part 2 training runs also save a `*_metadata.json` file next to each model
checkpoint with the command, seed, hyperparameters, and package versions.

## Report Assets

After experiments finish, collect report-ready tables, plots, and render images:

```bash
python make_report_assets.py
```

The PNG files are written to `report_assets/`, including summary tables,
comparison plots, and the required Part 2 result plots.

## Project Structure

```text
FAIML-RL-26/
├── README.md
├── make_report_assets.py
├── requirements.txt
├── renders/                          # start+finish render images (generated)
├── report_assets/                    # report-ready tables, plots, renders (generated)
├── part1/
│   ├── agent.py
│   ├── test_random_policy.py
│   ├── train.py
│   ├── evaluate_required_actor_critic_hopper.py
│   ├── render_required_actor_critic_hopper.py
│   └── render_trained_hopper.py
└── part2/
    ├── eval_sb3.py
    ├── plot_results.py
    ├── rand_wrapper.py
    ├── render_panda.py
    ├── train_sb3.py
    └── panda-gym/
```
