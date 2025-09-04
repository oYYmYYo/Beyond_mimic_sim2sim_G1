# Beyondmimic_sim2sim
Sim2sim based on Unitree_rl_gym.

# 基于Unitree_rl_gym搭的beyondmimic复现


需要修改输入的motion和policy。
请大家一起讨论哈，争取完善。
提供的模型是基于lafan训练的dance2_subject4。


# 注意
本代码基于的是beyondmimic作者开源算法中不带状态估计的训练配置，即Tracking-Flat-G1-Wo-State-Estimation-v0。
这种训练方式的状态空间维度为154维。
如果希望使用原本的配置来进行测试，请在anchorori的obs前增加三维度的相对位置，以及在angvel前增加三维度的根坐标系速度。

# Acknowledgement：
[1] Beyondmimic训练源码：https://github.com/HybridRobotics/whole_body_tracking

[2] Unitree_rl_gym仓库： https://github.com/unitreerobotics/unitree_rl_gym
