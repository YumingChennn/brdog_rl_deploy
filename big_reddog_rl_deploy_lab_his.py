import time
import sys
import numpy as np
import threading
import traceback
import yaml
import csv
import argparse
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

class Controller:
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config/big_reddog_lab_his.yaml", help="config file path")
        args, _ = parser.parse_known_args()
        config_file = args.config
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            policy_path = config["policy_path"]
            self.policy = torch.jit.load(policy_path)
            self.dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = np.array(config["action_scale"], dtype=np.float32)
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            one_step_obs_size = config["one_step_obs_size"]
            obs_buffer_size = config.get("obs_buffer_size", 1)

            max_lin = config.get("max_lin_vel", 1.0)
            max_ang = config.get("max_ang_vel", 1.0)
            
            # Transition steps for stand/sit (e.g., 1.2s / 0.005s = 240 steps)
            self.transition_steps = int(3 / self.dt)
        
        self.teleop = gui_teleop.GUITeleop(config_init=config["cmd_init"], max_lin=max_lin, max_ang=max_ang)
        self.target_dof_pos = self.default_angles.copy()
        self.target_dof_vel = np.zeros(num_actions)
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.obs_history_buffer = torch.zeros((obs_buffer_size, one_step_obs_size))
            
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

            # Joint position delta for all 12 joints
            pos_delta = (self.qpos - self.default_angles) * self.dof_pos_scale
            
            obs_list = [
                (self.ang_vel.copy() * self.ang_vel_scale).astype(np.float32),
                gravity_b.astype(np.float32),
                current_cmd_vel * self.cmd_scale,
                pos_delta.astype(np.float32),
                (self.qvel * self.dof_vel_scale).astype(np.float32),
                self.action.copy()
            ]

            obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]

            current_obs = torch.cat(obs_list, dim=0)

            self.obs_history_buffer = torch.roll(self.obs_history_buffer, shifts=-1, dims=0)
            self.obs_history_buffer[-1] = current_obs

            # Stack by feature group: [feat1_t, feat1_t-1, ..., feat2_t, ...]
            # feature_groups: ang_vel(3), gravity(3), cmd(3), pos(12), vel(12), action(12)
            split_sizes = [3, 3, 3, 12, 12, 12]
            feature_groups = []
            start_idx = 0
            
            for size in split_sizes:
                end_idx = start_idx + size
                # Extract column slice [start_idx:end_idx] from all timesteps, then flatten
                feature_slice = self.obs_history_buffer[:, start_idx:end_idx]
                feature_groups.append(feature_slice.flatten())
                start_idx = end_idx
            
            obs_tensor_buf = torch.cat(feature_groups).unsqueeze(0)
            obs_tensor_buf = torch.clip(obs_tensor_buf, -100, 100)

            # obs inference
            self.action = self.policy(obs_tensor_buf).detach().numpy().squeeze()

            # Apply actions to all 12 joints (position control only)
            for i in range(NUM_MOTORS):
                self.target_dof_pos[i] = self.default_angles[i] + self.action[i] * self.action_scale[i]
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