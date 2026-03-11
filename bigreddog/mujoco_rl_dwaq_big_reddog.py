import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from keyboard_controller import KeyboardController

# from gamepaded import gamepad_reader
NUM_MOTOR = 12


def get_activation(act_name: str) -> nn.Module | None:
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class ActorCritic_DWAQ(nn.Module):
    """Actor-Critic with DWAQ (Deep Variational Autoencoder for Walking) context encoder.
    
    The context encoder (β-VAE) infers velocity and latent state from observation history.
    Sim2Sim 版本只包含 Actor 和 Encoder 部分。
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cenet_in_dim: int,
        cenet_out_dim: int,
        obs_dim: int,
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.activation = get_activation(activation)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, num_actions)
        )

        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        # Decoder (for completeness, not used in inference)
        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, self.obs_dim)
        )

        print(f"[INFO] ActorCritic_DWAQ 初始化:")
        print(f"  - Actor 输入: {num_actor_obs} (obs + latent_code)")
        print(f"  - Encoder 输入: {cenet_in_dim} (obs_history)")
        print(f"  - Encoder 输出: {cenet_out_dim} (vel:3 + latent:{cenet_out_dim-3})")
        print(f"  - Actor 输出: {num_actions}")

    def cenet_forward(self, obs_history: torch.Tensor):
        """Forward pass through the context encoder (β-VAE).
        
        Args:
            obs_history: Flattened observation history [batch, history_len * obs_dim]
            
        Returns:
            code: Concatenated latent code [vel(3) + latent(16)] for actor
        """
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        # 推理时使用均值，不采样
        code = torch.cat((mean_vel, mean_latent), dim=-1)
        return code

    def act_inference(self, observations: torch.Tensor, obs_history: torch.Tensor):
        """Compute deterministic actions for inference.
        
        Args:
            observations: Current actor observations [batch, obs_dim]
            obs_history: Observation history for context encoder [batch, history_len * obs_dim]
            
        Returns:
            Mean actions (deterministic)
        """
        code = self.cenet_forward(obs_history)
        actor_input = torch.cat((code, observations), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

def calculate_com_in_base_frame(model, data, base_body_id):
    total_mass = 0.0
    com_sum = np.zeros(3)

    # Get base position and rotation
    base_pos = data.xipos[base_body_id]  # Position of the base in world coordinates
    base_rot = data.ximat[base_body_id].reshape(3, 3)  # Rotation matrix of the base

    for i in range(model.nbody):
        # Get body mass and world COM position
        mass = model.body_mass[i]
        world_com = data.xipos[i]

        # Transform COM to base coordinates
        local_com = world_com - base_pos  # Translate to base origin
        local_com = base_rot.T @ local_com  # Rotate into base frame

        # Accumulate mass-weighted positions
        com_sum += mass * local_com
        total_mass += mass

    # Compute overall COM in base coordinates
    center_of_mass_base = com_sum / total_mass
    return center_of_mass_base

def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    
    Args:
        q: The quaternion in (w, x, y, z) format. Shape is (..., 4).
        v: The vector in (x, y, z) format. Shape is (..., 3).
        
    Returns:
        The rotated vector in (x, y, z) format. Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Equivalent to (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded
    
    # Equivalent to torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0
    
    # Equivalent to the torch.bmm or torch.einsum operations
    # This calculates the dot product between q_vec and v
    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0
    
    return a - b + c

def get_gravity_orientation(quaternion):
    """
    Get the gravity vector in the robot's base frame.
    Uses the exact algorithm from your PyTorch code.
    
    Args:
        quaternion: Quaternion in (w, x, y, z) format.
        
    Returns:
        3D gravity vector in the robot's base frame.
    """
    # Ensure quaternion is a numpy array
    quaternion = np.array(quaternion)
    
    # Standard gravity vector in world frame (pointing down)
    gravity_world = np.array([0, 0, -1])
    
    # Handle both single quaternion and batched quaternions
    if quaternion.shape == (4,):
        quaternion = quaternion.reshape(1, 4)
        gravity_world = gravity_world.reshape(1, 3)
        result = quat_rotate_inverse(quaternion, gravity_world)[0]
    else:
        gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
        result = quat_rotate_inverse(quaternion, gravity_world)
    
    return result

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    keyboard = KeyboardController()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        
        # 加载策略网络 (使用 ActorCritic_DWAQ)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config["obs_buffer_size"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        
    # 创建 ActorCritic_DWAQ 模型
    cenet_in_dim = obs_buffer_size * one_step_obs_size  # 历史观测展平
    cenet_out_dim = config.get("cenet_out_dim", 19)  # velocity(3) + latent(16)
    num_actor_obs = one_step_obs_size + cenet_out_dim  # 当前观测 + latent code
        
    policy = ActorCritic_DWAQ(
        num_actor_obs=num_actor_obs,
        num_critic_obs=200,  # 不重要，推理时不用
        num_actions=num_actions,
        cenet_in_dim=cenet_in_dim,
        cenet_out_dim=cenet_out_dim,
        obs_dim=one_step_obs_size,
        activation='elu'
    )
        
    # 加载权重
    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
    # 只加载需要的权重
    needed_keys = ['actor', 'encoder', 'encode_mean_latent', 'encode_mean_vel',
                    'encode_logvar_latent', 'encode_logvar_vel']
    filtered_state_dict = {}
    for key, value in model_state_dict.items():
        for needed in needed_keys:
            if key.startswith(needed):
                filtered_state_dict[key] = value
                break
        
    missing, unexpected = policy.load_state_dict(filtered_state_dict, strict=False)
        
    if missing:
            print(f"[WARNING] 缺少的权重: {missing[:5]}...")
    if unexpected:
        print(f"[WARNING] 多余的权重: {unexpected[:5]}...")

    policy.eval()
    print(f"[INFO] 策略加载完成: {policy_path}")


    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = np.zeros((
        obs_buffer_size, one_step_obs_size),
        dtype=np.float32
    )

    # gamepad = gamepad_reader.Gamepad(vel_scale_x=0.5, vel_scale_y=0.5, vel_scale_rot=0.8)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1


    # Record data
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    action_list = []
    feet_list=[]
    
    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
    
            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)

            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            # com_base = calculate_com_in_base_frame(m, d, base_body_id)
            # print("Center of Mass in Base Coordinates:", com_base)

            counter += 1
            if counter % control_decimation == 0 and counter > 0:

                # create observation
                foot_ids = [4, 6, 8, 10]  
                foot_heights = [d.xipos[i][2] for i in foot_ids]  #get feethigh
                
                position = d.qpos[:3]
                qpos = d.sensordata[:12]
                qvel = d.sensordata[12:24]
                ang_vel_B = d.sensordata[40:43]
                imu_quat =d.sensordata[36:40]
                cmd_vel = np.array(config["cmd_init"], dtype=np.float32)
                
                # lin_vel_B = quat_rotate_inverse(imu_quat, lin_vel_I)
                # ang_vel_B = quat_rotate_inverse(imu_quat, ang_vel_I)

                gravity_b = get_gravity_orientation(imu_quat)
                
                cmd = keyboard.read()  # 读取键盘输入，更新 cmd

                obs_list = [
                    ang_vel_B * ang_vel_scale,
                    gravity_b,
                    cmd * cmd_scale,
                    (qpos - default_angles) * dof_pos_scale,
                    qvel * dof_vel_scale,
                    action.astype(np.float32)
                ]
                ## Record Data ##
                ang_vel_data_list.append(ang_vel_B * ang_vel_scale)
                gravity_b_list.append(gravity_b)
                joint_vel_list.append(qvel * dof_vel_scale)
                action_list.append(action)
                feet_list.append(foot_heights)
               
                # 构建当前单步观测
                current_obs = np.concatenate(obs_list, axis=0).astype(np.float32)
                
                # 更新观测历史 (FIFO)
                obs_history = np.roll(obs_history, shift=-1, axis=0)
                obs_history[-1] = current_obs.copy()
                
                # 转换为 tensor
                current_obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)
                obs_history_flat = obs_history.flatten()
                obs_history_tensor = torch.tensor(obs_history_flat, dtype=torch.float32).unsqueeze(0)
                
                # 使用 act_inference 进行策略推理
                with torch.no_grad():
                    action_tensor = policy.act_inference(current_obs_tensor, obs_history_tensor)
                
                action = action_tensor.squeeze(0).numpy()

                # transform action to target_dof_pos
                if counter < 300:
                    target_dof_pos = default_angles
                else:
                    target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = action * action_scale + default_angles
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Plot the collected data after the simulation ends
    plt.figure(figsize=(18, 20))

    plt.subplot(4, 2, 1)
    for i in range(3): 
        plt.plot([step[i] for step in lin_vel_data_list], label=f"Linear Velocity {i}")
    plt.title(f"History Linear Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(4, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(4, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in gravity_b_list], label=f"Project Gravity {i}")
    plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(4, 2, 5)
    for i in range(2):
        plt.plot([step[i] for step in joint_vel_list], label=f"Joint Velocity {i}")
    plt.title(f"History Joint Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(4, 2, 6)
    for i in range(2):
        plt.plot([step[i] for step in action_list], label=f"velocity Command {i}")
    plt.title(f"History Torque Command", fontsize=10, pad=10)  # Added pad for spacing
    plt.subplot(4, 2, 7)
    Footname=["FL","RL","FR","RR"]
    for i in range(len(feet_list[0])):
        plt.plot([step[i] for step in feet_list], label=f"{Footname[i]} Height")# 0FL 1FR 2RL 3RR
    plt.title("Foot Height (z)", fontsize=10, pad=10)
    plt.legend()
    plt.tight_layout()
    plt.show()