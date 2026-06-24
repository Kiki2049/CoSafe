# CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving

In this work, we propose **CoSafe**, a modular and ADS-agnostic cooperative runtime safety assurance framework for mixed-traffic autonomous driving to extend runtime safety assurance from ego-centric takeover to scene-level cooperative intervention. CoSafe leverages vehicle-to-everything (V2X) communication to enable roadside units (RSUs) to monitor scene-level hazards and jointly mitigate hazards involving multiple vehicles.

![Overview](assets/Overview.png)

This work has been submitted to ICSE 2027.

## Benchmark

[InterDrive](https://github.com/cxliu0314/CoLMDriver) is a closed-loop benchmark built on the V2Xverse platform that systematically assesses multi-vehicle interaction in CARLA simulator. It comprises 10 typical multi-vehicle scenario types with reference to the pre-crash typology of the NHTSA, organized into three categories, InterDrive-IC where vehicles enter, meet, and exit an intersection from different directions, InterDrive-LM where vehicles merge into the same lane across parallel lanes, T-junctions, and highway ramps, and InterDrive-LC where vehicles traveling in parallel execute lane-change maneuvers that intersect adjacent vehicles' trajectories. The routes are stored [here](simulation/leaderboard/data/Interdrive).

## Target ADSs and Checkpoints

We integrate CoSafe with seven state-of-the-art ADSs spanning five families to evaluate its applicability across heterogeneous ADSs. All models use their original checkpoints, or are trained with the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) training set if original checkpoints are not available. The target ADSs are:
* a modular ADS, [Autoware](https://github.com/TUMFTM/autoware_carla_leaderboard?tab=readme-ov-file);
* three end-to-end ADSs, [TCP](https://github.com/OpenDriveLab/TCP), [UniAD](https://github.com/OpenDriveLab/UniAD), and [VAD](https://github.com/hustvl/VAD);
* an LLM-driven ADS, [LMDrive](https://github.com/opendilab/LMDrive);
* a shared-perception cooperative ADS, [CoDriving](https://github.com/CollaborativePerception/V2Xverse);
* an LLM-mediated negotiation cooperative ADS, [CoLMDriver](https://github.com/cxliu0314/CoLMDriver).

In our prototype, all target ADSs share a single auxiliary BEV encoder and
RSU-side fusion model, both pre-trained on [DeepAccident](https://github.com/tianqi-wang1996/DeepAccident):

* [CoBEVT](https://github.com/DerrickXuNu/CoBEVT);
* [BEVerse](https://github.com/zhangyp15/BEVerse).

## Experiment Environment
We recommend and conduct all the experiments on a system with the following specifications:

* Operating System: Ubuntu 20.04.4 LTS
* GPU: 4 NVIDIA GeForce RTX 3090 GPUs
* CPU: Intel(R) Xeon(R) Silver 4310 @ 2.10GHz
* Memory: 128GB

## Installation

### vllm env (only for the CoLMDriver agent)
Refer to the official repo of [vllm](https://github.com/vllm-project/vllm).

### CoSafe env

#### Step 1: Basic environment
```Shell
git clone <anonymized-CoSafe-repo>
cd CoSafe
conda create --name cosafe python=3.7 cmake=3.22.1
conda activate cosafe
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

pip install -r opencood/requirements.txt
pip install -r simulation/requirements.txt
```

#### Step 2: Download and set up CARLA 0.9.10.1
```Shell
chmod +x simulation/setup_carla.sh
./simulation/setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
mkdir external_paths
ln -s ${PWD}/carla/ external_paths/carla_root
# If you already have a CARLA, just create a soft link to external_paths/carla_root
```
The file structure should be:
```Shell
|--CoSafe
    |--external_paths
        |--carla_root
            |--CarlaUE4
            |--PythonAPI
            |--CarlaUE4.sh
            ...
```
Note: setuptools==41 provides the `easy_install` feature. After installing the carla egg you can upgrade setuptools again to avoid `No module named distutils_hack`.

#### Step 3: Install Spconv (1.2.1)
We use spconv 1.2.1 to generate voxel features in perception module. 

To install it, follow the guide in [spconv](https://github.com/traveller59/spconv/tree/v1.2.1).

#### Step 4: Set up OpenCOOD
```Shell
python setup.py develop

# Bbox IoU cuda compile
python opencood/utils/setup.py build_ext --inplace
```

#### Step 5: Install pypcd
```Shell
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ../CoSafe
```

#### Step 6: Build the CoSafe Cython kernels
The CoSafe middleware kernels are compiled from `cs_cosafe_c/kernels/*.pyx`:
```Shell
pip install cython numpy
python cs_cosafe_c/setup.py build_ext --inplace
```

#### Step 7: Build mmcv_b2d for the Bench2Drive agents
UniAD / VAD / TCP run on a customized MMCV (`mmcv_b2d`, renamed so it coexists with the perception MMCV):
```Shell
# GCC 9.4 is strongly recommended.
cd Bench2DriveZoo
export PATH=YOUR_GCC_PATH/bin:$PATH
export CUDA_HOME=YOUR_CUDA_PATH/
pip install ninja packaging
pip install -v -e .
cd ..
```

#### Step 8: Set up the cooperative-perception model (BEVerse V2X)
The RSU-side cooperative perception uses the DeepAccident BEVerse V2X model. Install its inference dependencies and `mmdet3d` into the same `cosafe` env:
```Shell
pip install -r DeepAccident/requirements.txt
cd DeepAccident && python setup.py develop && cd ..
```
This installs the standard `mmcv-full==1.3.16` / `mmdet` / `mmseg` / `mmdet3d`, which coexist with the `mmcv_b2d` from Step 7.

Then download the BEVerse V2X checkpoint from [Google drive](https://drive.google.com/file/d/1B0zxVzp9QElb2T2HRbrnkmJ7fYylj-fb) and place it as follows:
```Shell
|--CoSafe
    |--DeepAccident
        |--work_dirs
            |--DeepAccident_tiny_v2x_4agents_infra_BEVT
                |--latest.pth
```

## Evaluation

Each ADS is evaluated in up to three configurations: the native ADS (baseline), with Argus (ego-centric safety assurance framework), and with CoSafe. The `run_parallel*` scripts launch CARLA and the eval workers automatically — by default CARLA on GPU 0 (ports 2000/2010/2020) and three eval workers on GPUs 1–3, splitting the 46 routes into three groups. Edit the group/GPU/port settings at the top of each script for your machine.

### Evaluation of TCP, UniAD, VAD, and CoDriving
**Step 1:** Download checkpoints from [Google drive](https://drive.google.com/file/d/1B0zxVzp9QElb2T2HRbrnkmJ7fYylj-fb). The downloaded checkpoints of these ADSs should follow this structure:

```Shell
|--CoSafe
    |--ckpt
        |--TCP
        |--uniad
        |--vad
        |--codriving
```

**Step 2:** Run the evaluation scripts:

```Shell
conda activate cosafe

# ADS alone (baseline)
bash scripts/eval/run_parallel.sh <method>

# ADS + Argus  (tcp / uniad / vad)
bash scripts/eval/run_parallel.sh <method>_mr

# ADS + CoSafe
bash scripts/eval/run_parallel_cosafe.sh <method>
```
`<method>` is one of `tcp`, `uniad`, `vad`, `codriving`. 

### Evaluation of LMDrive
LMDrive is served as a separate FastAPI inference process. The model code (LAVIS) and checkpoint paths are configured in `simulation/leaderboard/team_code/agent_config/lmdriver_config_8_10.py`.

**Step 1:** Download the weights and the model code (LAVIS) from the official [LMDrive](https://github.com/opendilab/LMDrive) repo, and place them following this structure:
```Shell
|--CoSafe
    |--simulation
        |--assets
            |--LLaVA
                |--ckpt
                    |--llava-v1.5-7b
            |--LMDrive
                |--ckpt
                    |--vision-encoder-r50.pth.tar
                    |--llava-v1.5-checkpoint.pth
                |--LAVIS
```

**Step 2:** Start the LMDrive inference server (default port 8100) on a GPU with free memory:
```Shell
conda activate cosafe
CUDA_VISIBLE_DEVICES=<gpu> python simulation/leaderboard/team_code/lmdrive_server.py \
    --config simulation/leaderboard/team_code/agent_config/lmdriver_config_8_10.py \
    --port 8100
```

**Step 3:** In another terminal, run the evaluation. The agent reaches the server through `LMDRIVE_SERVER_URL` (default `http://localhost:8100`):
```Shell
conda activate cosafe
export LMDRIVE_SERVER_URL=http://localhost:8100

# LMDrive alone (baseline)
bash scripts/eval/run_parallel.sh lmdrive

# LMDrive + CoSafe
bash scripts/eval/run_parallel_cosafe.sh lmdrive
```

### Evaluation of CoLMDriver

**Step 1:** Download checkpoints from [Google drive](https://drive.google.com/file/d/1z3poGdoomhujCNQtoQ80-BCO34GTOLb-/view). The downloaded checkpoints of CoLMDriver should follow this structure:
```Shell
|--CoSafe
    |--ckpt
        |--colmdriver
            |--LLM
            |--perception
            |--VLM
            |--waypoints_planner
```

**Step 2:** CoLMDriver additionally needs its VLM and LLM served via vLLM. In the `vllm` env:
```Shell
conda activate vllm
# VLM
CUDA_VISIBLE_DEVICES=0 vllm serve ckpt/colmdriver/VLM --port 1111 --max-model-len 8192 --trust-remote-code --enable-prefix-caching
# LLM
CUDA_VISIBLE_DEVICES=1 vllm serve ckpt/colmdriver/LLM --port 8888 --max-model-len 4096 --trust-remote-code --enable-prefix-caching
```
If you use other ports, update `comm_client` / `vlm_client` in `simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml` accordingly. 

**Step 3:** Run the evaluation scripts:
```Shell
conda activate cosafe

# CoLMDriver alone (baseline)
bash scripts/eval/run_parallel.sh colmdriver

# CoLMDriver + CoSafe
bash scripts/eval/run_parallel_cosafe.sh colmdriver
```

### Evaluation of Autoware
Autoware runs in Docker (a ROS 2 bridge plus an Autoware Universe instance per ego) defined under `docker/autoware/`.

**Step 1:** Prepare the two Docker images and the maps.
- `cosafe_autoware_bridge:latest` — the ROS 2 bridge, built from `docker/autoware/bridge/Dockerfile`:
  ```Shell
  cd docker/autoware && docker compose build bridge_0 && cd ../..
  ```
- `autoware_universe:carla-priv` — Autoware Universe 1.7.1 with the CARLA `audi_etron` vehicle model. Build it by following the "Autoware universe" section of [TUM autoware_carla_leaderboard](https://github.com/TUMFTM/autoware_carla_leaderboard).
- lanelet2 maps — download from the [TUM map release](https://github.com/TUMFTM/autoware_carla_leaderboard/releases) and unzip into `docker/autoware/maps/Town05`, `docker/autoware/maps/Town07`, etc. (mounted into the containers as `/maps`).

**Step 2:** Run the evaluation. Each script starts CARLA and manages the Autoware containers automatically:
```Shell
conda activate cosafe

# Autoware alone (baseline)
bash scripts/eval/run_baseline_autoware.sh

# Autoware + CoSafe
bash scripts/eval/run_cosafe_autoware.sh
```

### Evaluation of Ablation
CoSafe's components can be ablated on TCP using the script `scripts/eval/run_parallel_cosafe_ablation.sh`. Three variants are available: `no_coop` (disable cooperative perception, RSU-only BEV), `no_risk` (disable the deadlock oracle), and `no_joint` (mitigate with per-AV greedy selection).

### Evaluation of Mixed-Deployment
Each scenario mixes TCP, LMDrive, and UniAD. Start the LMDrive inference server first (see [Evaluation of LMDrive](#evaluation-of-lmdrive)) and run the evaluation scripts:

```Shell
conda activate cosafe
export LMDRIVE_SERVER_URL=http://localhost:8100

# Mixed deployment (baseline)
bash scripts/eval/run_parallel_mixed.sh baseline

# Mixed deployment + CoSafe
bash scripts/eval/run_parallel_mixed.sh cosafe
```

### Results

Results are saved to `results/results_driving_<method>` (baseline), `results/results_driving_<method>_mr` (Argus), and `results/results_driving_cosafe_<method>` (CoSafe). 

Summarize any run with:
```Shell
python visualization/result_analysis_csv.py <results_dir>
```

## Training of Ego-level Candidate Generation

The ego-level candidate generator is trained on the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) expert dataset.

**Step 1: K-means trajectory anchors.** Cluster expert ego trajectories into 128 anchors:
```Shell
conda activate cosafe
python cs_cosafe_c/training/cs_kmeans_trajectory.py \
    --pkl_path Bench2DriveZoo/data/infos/b2d_e2e_ego_infos_train.pkl \
    --output_dir cs_cosafe_c/training/kmeans_output_base/k128 \
    --num_anchors 128 --t_plan 20 --dt_target 0.1
```

**Step 2: Extract BEV features.** Run the BEVerse V2X encoder over the dataset to cache per-frame BEV features:
```Shell
python cs_cosafe_c/training/cs_extract_bev.py \
    --pkl_path Bench2DriveZoo/data/infos/b2d_e2e_ego_infos_train.pkl \
    --data_root <bench2drive_data_root> \
    --config_path DeepAccident/projects/configs/DeepAccident_tiny_v2x_4agents_infra.py \
    --checkpoint DeepAccident/work_dirs/DeepAccident_tiny_v2x_4agents_infra_BEVT/latest.pth \
    --output_dir cs_cosafe_c/training/bev_features
```

**Step 3: Train the anchor-residual network.** Train the offset and score heads on the cached features and anchors:
```Shell
python cs_cosafe_c/training/cs_wote_train.py \
    --pkl_path Bench2DriveZoo/data/infos/b2d_e2e_ego_infos_train.pkl \
    --anchors_path cs_cosafe_c/training/kmeans_output_base/k128/trajectory_anchors_128.npy \
    --bev_dir cs_cosafe_c/training/bev_features \
    --output_dir cs_cosafe_c/training/output
```
The best checkpoint is written to `cs_cosafe_c/training/output/best.pth`.

## Acknowledgments
We thank the authors of the following repositories for their contributions:
* [CoLMDriver](https://github.com/cxliu0314/CoLMDriver)
* [DeepAccident](https://github.com/tianqi-wang1996/DeepAccident)
* [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)