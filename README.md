# CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving

In this work, we propose **CoSafe**, a modular and ADS-agnostic cooperative runtime safety assurance framework for mixed-traffic autonomous driving to extend runtime safety assurance from ego-centric takeover to scene-level cooperative intervention. CoSafe leverages vehicle-to-everything (V2X) communication to enable roadside units (RSUs) to monitor scene-level hazards and jointly mitigate hazards involving multiple vehicles.

![Overview](assets/Overview.png)

This work has been submitted to ICSE 2027.

## Benchmark

### InterDrive

[InterDrive](https://github.com/cxliu0314/CoLMDriver) is a closed-loop benchmark built on the V2Xverse platform that systematically assesses multi-vehicle interaction in CARLA simulator. It comprises 10 typical multi-vehicle scenario types with reference to the pre-crash typology of the NHTSA, organized into three categories, InterDrive-IC where vehicles enter, meet, and exit an intersection from different directions, InterDrive-LM where vehicles merge into the same lane across parallel lanes, T-junctions, and highway ramps, and InterDrive-LC where vehicles traveling in parallel execute lane-change maneuvers that intersect adjacent vehicles' trajectories. The routes are stored in [there](simulation/leaderboard/data/Interdrive).

## Target ADSs and Checkpoints

## Experiment Environment
We recommend and conduct all the experiments on a system with the following specifications:

* Operating System: Ubuntu 20.04.4 LTS
* GPU: 4 NVIDIA GeForce RTX 3090 GPUs
* CPU: Intel(R) Xeon(R) Silver 4310 @ 2.10GHz
* Memory: 128GB

## Installation
