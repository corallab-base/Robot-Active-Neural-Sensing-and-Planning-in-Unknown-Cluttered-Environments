Robot Active Neural Sensing and Planning in Unknown Cluttered Environments
Code for [Robot Active Neural Sensing and Planning in Unknown Cluttered Environments](https://ieeexplore.ieee.org/document/10101696)
```Bibtex
@ARTICLE{10101696,
  author={Ren, Hanwen and Qureshi, Ahmed H.},
  journal={IEEE Transactions on Robotics},
  title={Robot Active Neural Sensing and Planning in Unknown Cluttered Environments},
  year={2023},
  volume={39},
  number={4},
  pages={2738-2750},
  keywords={Robots;Sensors;Cameras;Planning;Robot vision systems;Visualization;Manipulators;Active sensing;deep learning;planning and control;scene reconstruction;unknown environments},
  doi={10.1109/TRO.2023.3262114}}
```
---
## Dependencies
- environment.yml 
- Isaac Gym https://developer.nvidia.com/isaac-gym
- OMPL https://ompl.kavrakilab.org/ 
- trac-IK https://bitbucket.org/traclabs/trac_ik/src/master/ 
- PoinTr https://github.com/yuxumin/PoinTr
---

## Pretrained Models and usage
- [ScoreNet](https://drive.google.com/drive/folders/1G3CgzlIclktMbbc8krXBYMFIZ6Gd1v_E?usp=sharing)\
- [VPFormer](https://drive.google.com/drive/folders/1UBZgqJjepLXYmBXpAqzX179p1rn83Vpg?usp=sharing) \
- [PoinTr](https://drive.google.com/drive/folders/1K_VioYllsd5OuU0rB5PZnIlrKHjG-a76?usp=sharing)   \
---

## Usage
Under sim folder:
`ur5e_refactor_gen_data.py` is used to generate robot perception dataset to train the ScoreNet located in `ScoreNet/`\
`ur5e_refactor_get_table.py` is used to evaluate the performance of different active sensing methods \
`ur5e_refactor_real_exp.py` is used to conduct real experiment, it will stop and wait for real-world RGB-D and segmentation images before reconstruction
