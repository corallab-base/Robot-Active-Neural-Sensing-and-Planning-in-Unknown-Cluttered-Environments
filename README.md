Robot Active Neural Sensing and Planning in Unknown Cluttered Environments
---
## Dependencies
- environment.yml 
- Isaac Gym https://developer.nvidia.com/isaac-gym
- OMPL https://ompl.kavrakilab.org/ 
- trac-IK https://bitbucket.org/traclabs/trac_ik/src/master/ 
- PoinTr https://github.com/yuxumin/PoinTr
---

## Pretrained Models and usage
---

## Usage
Under sim folder:
`ur5e_refactor_gen_data.py` is used to generate robot perception dataset to train the ScoreNet located in `ScoreNet/`
`ur5e_refactor_get_table.py` is used to evaluate the performance of different active sensing methods
`ur5e_refactor_real_exp.py` is used to conduct real experiment, it will stop and wait for real-world RGB-D and segmentation images before reconstruction
