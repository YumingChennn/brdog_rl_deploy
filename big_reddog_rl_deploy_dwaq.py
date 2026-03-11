import time
import sys
import numpy as np
import threading
import traceback
import yaml
import csv
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt # Import for plotting
import torch
import gui_teleop 

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_MOTORS = 12

class Filter:
    def __init__(self, alpha):
        self.filter_value = None
        self.alpha = alpha
    
    def filt(self, input):
        if self.filter_value is None:
            self.filter_value = input
        else:
            self.filter_value = self.alpha * input + (1 - self.alpha) * self.filter_value
        return self.filter_value

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


class Controller:
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config/big_reddog_dwaq.yaml", help="config file path")
        args, _ = parser.parse_known_args()
        config_file = args.config
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            policy_path = config["policy_path"]
            self.dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            one_step_obs_size = config["one_step_obs_size"]
            obs_buffer_size = config["obs_buffer_size"]

            max_lin = config.get("max_lin_vel", 1.0)
            max_ang = config.get("max_ang_vel", 1.0)
            
            # Transition steps for stand/sit (e.g., 1.2s / 0.005s = 240 steps)
            self.transition_steps = int(3 / self.dt)
            
            # 创建 DWAQ 策略模型
            cenet_in_dim = obs_buffer_size * one_step_obs_size
            cenet_out_dim = config.get("cenet_out_dim", 19)  # vel(3) + latent(16)
            num_actor_obs = one_step_obs_size + cenet_out_dim
            
            self.policy = ActorCritic_DWAQ(
                num_actor_obs=num_actor_obs,
                num_critic_obs=200,  # 推理时不使用
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
            
            missing, unexpected = self.policy.load_state_dict(filtered_state_dict, strict=False)
            
            if missing:
                print(f"[WARNING] 缺少的权重: {missing[:5]}...")
            if unexpected:
                print(f"[WARNING] 多余的权重: {unexpected[:5]}...")
            
            self.policy.eval()
            print(f"[INFO] DWAQ 策略加载完成: {policy_path}")
        
        self.teleop = gui_teleop.GUITeleop(config_init=config["cmd_init"], max_lin=max_lin, max_ang=max_ang)
        self.target_dof_pos = self.default_angles.copy()
        self.target_dof_vel = np.zeros(num_actions)
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.obs_history_buffer = np.zeros((obs_buffer_size, one_step_obs_size), dtype=np.float32)
            
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.controller_rt = 0.0
        self.is_running = False
        self.counter = 0
        self.step = 0
        self.current_pos = None

        self.ang_vel_data = []
        self.qtau_data = []
        self.qtau_cmd = []

        # Data logging lists for plotting (for motor 0 as an example)
        self.time_data = []
        self.qpos_data = []
        self.qvel_data = []
        self.qtau_data = []
        self.dq_cmd_data = []
        self.tau_cmd_data = []


        # thread handling
        self.lowCmdWriteThreadPtr = None

        # state
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.quat = np.zeros(4) # q_w q_x q_y q_z
        self.ang_vel = np.zeros(3)

        self.mode = ''
        # self.dt = 0.001
        self.start_time = time.perf_counter() # To calculate elapsed time


        self.crc = CRC()

        self.first_logged = False
        self.second_logged = False
        
        # For frequency measurement
        self.last_policy_time = None
        self.policy_call_count = 0

    # Control methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # Init default pos #
        self.Start()
        self.start_time = time.perf_counter() # Reset start time after threads are initialized

        print("Initial Sucess !!!")

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    

    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite, daemon=True)
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        self.teleop.close()
        if self.lowCmdWriteThreadPtr:
            self.lowCmdWriteThreadPtr.join()


    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.get_current_state()
        # self.record_data() # Record data when a new state message is received
        # print(f'FL qpos {self.low_state.motor_state[0].q} FR qpos {self.low_state.motor_state[1].q} RL qpos {self.low_state.motor_state[2].q} RR qpos {self.low_state.motor_state[3].q}')
        # quat = self.low_state.imu_state.quaternion
        # ang_vel = self.low_state.imu_state.gyroscope
        # print(f'quat w: {self.quat[0]} x: {self.quat[1]} y: {self.quat[2]} z: {self.quat[3]}')
        # print(f'ang_vel x: {self.ang_vel[0]} y: {self.ang_vel[1]} z: {self.ang_vel[2]}')

    def reset_timer(self):
        self.controller_rt = 0.0
        self.counter = 0
        self.step = 0
        self.current_pos = self.qpos.copy()

    def sit(self):
        if self.step < self.transition_steps:
            phase = float(self.step) / float(self.transition_steps)
            for i in range(NUM_MOTORS):
                target_pos = self.current_pos[i] * (1 - phase) + self.sit_angles[i] * phase
                self.low_cmd.motor_cmd[i].q = target_pos
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.step += 1
        else:
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = self.sit_angles[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0

    def stand(self):
        if self.step < self.transition_steps:
            phase = float(self.step) / float(self.transition_steps)
            for i in range(NUM_MOTORS):
                target_pos = self.current_pos[i] * (1 - phase) + self.default_angles[i] * phase
                self.low_cmd.motor_cmd[i].q = target_pos
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.step += 1
        else:
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = self.default_angles[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def move(self):
        if self.counter % self.control_decimation == 0 and self.counter > 0:
            
            # Measure policy frequency
            current_time = time.perf_counter()
            if self.last_policy_time is not None:
                dt_policy = current_time - self.last_policy_time
                freq = 1.0 / dt_policy
                self.policy_call_count += 1
                if self.policy_call_count % 50 == 0:  # Print every 50 calls (~1 second)
                    print(f"Policy frequency: {freq:.2f} Hz (dt: {dt_policy*1000:.2f} ms)")
            self.last_policy_time = current_time

            current_cmd_vel = self.teleop.get_command()

            gravity_b = self.get_gravity_orientation(self.quat)
            
            obs_list = [
                self.ang_vel.copy() * self.ang_vel_scale,
                gravity_b,
                current_cmd_vel * self.cmd_scale,
                (self.qpos - self.default_angles) * self.dof_pos_scale,
                self.qvel * self.dof_vel_scale,
                self.action.copy()
            ]

            # 构建当前单步观测
            current_obs = np.concatenate(obs_list, axis=0).astype(np.float32)
            
            # 更新观测历史 (FIFO)
            self.obs_history_buffer = np.roll(self.obs_history_buffer, shift=-1, axis=0)
            self.obs_history_buffer[-1] = current_obs.copy()
            
            # 转换为 tensor
            current_obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)
            obs_history_flat = self.obs_history_buffer.flatten()
            obs_history_tensor = torch.tensor(obs_history_flat, dtype=torch.float32).unsqueeze(0)
            
            # 使用 DWAQ 的 act_inference 进行策略推理
            with torch.no_grad():
                action_tensor = self.policy.act_inference(current_obs_tensor, obs_history_tensor)
            
            self.action = action_tensor.squeeze(0).numpy()

            # Apply actions to all 12 joints (position control only)
            for i in range(NUM_MOTORS):
                self.target_dof_pos[i] = self.default_angles[i] + self.action[i] * self.action_scale
                self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0

            if not self.first_logged:
                print("First action command sent: ", time.time())
                self.first_logged = True
            if self.first_logged and not self.second_logged and self.qvel[3]>1.0:
                print("Second action command sent (robot starts moving): ", time.time())
                self.second_logged = True
        self.counter += 1
    
    def stand_up(self):
        self.mode = 'stand'
        self.reset_timer()

    def sit_down(self):
        self.mode = 'sit'
        self.reset_timer()
    
    def move_rl(self):
        self.mode = 'move'
        self.reset_timer()


    def get_current_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
            self.qtau[i] = self.low_state.motor_state[i].tau_est

        for i in range(3):
            self.ang_vel[i] = self.low_state.imu_state.gyroscope[i]
        
        # print("angular vel: ", self.ang_vel)

        for i in range(4):
            self.quat[i] = self.low_state.imu_state.quaternion[i]
    

    def record_data(self):
        """Records current state and command data for plotting."""
        # Use motor 0 for plotting as an example
        motor_idx = 3
        
        current_time = time.perf_counter() - self.start_time
        
        self.time_data.append(current_time)
        self.qpos_data.append(self.qpos[motor_idx])
        self.qvel_data.append(self.qvel[motor_idx])
        self.qtau_data.append(self.qtau[motor_idx])
        
        # Record command for the active motor
        self.dq_cmd_data.append(self.low_cmd.motor_cmd[motor_idx].dq)
        self.tau_cmd_data.append(self.low_cmd.motor_cmd[motor_idx].tau)



    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'stand':
                self.stand()
            elif self.mode == 'sit':
                self.sit()
            elif self.mode == 'move':
                self.move()
            
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    # if len(sys.argv)>1:
    #     ChannelFactoryInitialize(1, sys.argv[1])
    # else:
    #     ChannelFactoryInitialize(1, "lo") # default DDS port for pineapple
    # ChannelFactoryInitialize(1, "enx7cc2c65314b0")
    # ChannelFactoryInitialize(1, "lo")
    # ChannelFactoryInitialize(1, sys.argv[1])
    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    controller = Controller()
    controller.Init()

    command_dict = {
        "stand": controller.stand_up,
        "sit": controller.sit_down,
        "move": controller.move_rl,
    }

    while True:        
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break

        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(-1)