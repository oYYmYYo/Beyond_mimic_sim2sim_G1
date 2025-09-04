"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.",default = '/home/czhpc/Humanoid/LAFAN1_Retargeting_Dataset/g1/dance2_subject5.csv')
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")

parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.", default = "LAFAN_dance2_subject5")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
# 修改frame_range参数的默认值
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    # default=,  # 添加默认值
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,  # 添加这个参数
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range  # 保存frame_range
        self._load_motion()
        self._interpolate_motion_startend(50, 50)  # 首尾插值平滑
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            # 读取指定范围的帧
            start_frame, end_frame = self.frame_range
            # 注意：CSV文件通常第一行是标题，如果您的CSV有标题行，需要skiprows=1
            # 这里假设CSV没有标题行，直接从数据开始
            # 由于frame_range是从1开始计数的，而numpy是从0开始，所以需要调整
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=start_frame - 1,  # 跳过前面的帧
                    max_rows=end_frame - start_frame + 1,  # 读取指定范围的帧
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")


    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )
    def _interpolate_motion_startend(self, start_frame: int, end_frame: int):
        import numpy as np

        # 默认姿态
        default_pose = [0, 0, 0.76,
                        1, 0, 0, 0, 
                        -0.312, 0., 0., 0.669, -0.363, 0., -0.312, 0.,
                        0., 0.669, -0.363, 0., 0., 0., 0., 0.2,
                        0.2, 0., 0.6, 0., 0., 0., 0.2, -0.2,
                        0., 0.6, 0., 0., 0.]  # default pose
        default_p = np.array(default_pose[0:3])
        default_r = np.array(default_pose[3:7])
        default_dof = np.array(default_pose[7:])

        # 原始数据
        base_pos = self.motion_base_poss_input.cpu().numpy()
        base_rot = self.motion_base_rots_input.cpu().numpy()
        dof_pos = self.motion_dof_poss_input.cpu().numpy()

        # 首部插值 - 从默认姿态到第一帧
        if start_frame > 0:
            start_base_pos = np.linspace(default_p, base_pos[0], start_frame, endpoint=False)
            start_base_rot = np.zeros((start_frame, 4))
            for i in range(start_frame):
                start_base_rot[i] = quat_slerp(
                    torch.tensor(default_r, dtype=torch.float32), 
                    torch.tensor(base_rot[0], dtype=torch.float32), 
                    i / start_frame
                ).numpy()
            start_dof_pos = np.linspace(default_dof, dof_pos[0], start_frame, endpoint=False)
        else:
            start_base_pos = np.empty((0, 3))
            start_base_rot = np.empty((0, 4))
            start_dof_pos = np.empty((0, dof_pos.shape[1]))

        # 尾部插值
        if end_frame > 0:
            end_base_pos = np.linspace(base_pos[-1], base_pos[-1], end_frame + 1)[1:]
            end_base_rot = np.zeros((end_frame, 4))
            for i in range(end_frame):
                end_base_rot[i] = quat_slerp(torch.tensor(base_rot[-1], dtype=torch.float32), 
                                             torch.tensor(base_rot[-1], dtype=torch.float32), 
                                             (i + 1) / (end_frame + 1)).numpy()
            end_dof_pos = np.linspace(dof_pos[-1], default_dof, end_frame + 1)[1:]
        else:
            end_base_pos = np.empty((0, 3))
            end_base_rot = np.empty((0, 4))
            end_dof_pos = np.empty((0, dof_pos.shape[1]))

        # 合并数据
        new_base_pos = np.vstack([start_base_pos, base_pos, end_base_pos])
        new_base_rot = np.vstack([start_base_rot, base_rot, end_base_rot])
        new_dof_pos = np.vstack([start_dof_pos, dof_pos, end_dof_pos])

        # 转回torch
        self.motion_base_poss_input = torch.from_numpy(new_base_pos).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(new_base_rot).to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(new_dof_pos).to(self.device)
        self.input_frames = new_base_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"After interpolation: total frames: {self.input_frames}")


    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,  # 确保传递这个参数
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos.to(torch.float32)
        joint_vel[:, robot_joint_indexes] = motion_dof_vel.to(torch.float32)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            # 
            output_filename = f"./motions/{args_cli.output_name}.npz"
            # 
            import os
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            np.savez(output_filename, **log)
            print(f"[INFO]: Motion saved to: {output_filename}")

            # import wandb

            # COLLECTION = args_cli.output_name
            # run = wandb.init(project="csv_to_npz", name=COLLECTION)#,mode="offline"
            # print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            # REGISTRY = "motions"
            # logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            # run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            # print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")


def main():
    """Main function."""
    import os
    # os.environ["WANDB_ENTITY"] = "ustc"
    # os.environ["WANDB_MODE"] = "offline"  # Disable wandb online mode
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # parser = argparse.ArgumentParser(description="Convert CSV motion data to NPZ format.")
    # parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file",default = '/home/czhpc/Humanoid/LAFAN1_Retargeting_Dataset/g1/dance2_subject5.csv')
    # parser.add_argument("--input_fps", type=int, default=30, help="Input frame rate (default: 30)")
    # parser.add_argument("--output_name", type=str, required=True, help="Base name for output NPZ file", default = "LAFAN_dance2_subject5")
    # parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    
    args = parser.parse_args()
    # close sim app
    simulation_app.close()
