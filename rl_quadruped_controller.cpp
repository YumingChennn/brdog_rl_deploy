#include "rl_quadruped_controller/rl_quadruped_controller.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <yaml-cpp/yaml.h>

namespace rl_quadruped_controller
{

RLQuadrupedController::RLQuadrupedController() = default;
using config_type = controller_interface::interface_configuration_type;

controller_interface::InterfaceConfiguration RLQuadrupedController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

  conf.names.reserve(joint_names_.size() * command_interface_types_.size());
  for (const auto& joint_name : joint_names_)
  {
      for (const auto& interface_type : command_interface_types_)
      {
          if (!command_prefix_.empty())
          {
              conf.names.push_back(command_prefix_ + "/" + joint_name + "/" + interface_type);
          }
          else
          {
              conf.names.push_back(joint_name + "/" + interface_type);
          }
      }
  }

  return conf;
}

controller_interface::InterfaceConfiguration RLQuadrupedController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

  conf.names.reserve(joint_names_.size() * state_interface_types_.size());
  for (const auto& joint_name : joint_names_)
  {
      for (const auto& interface_type : state_interface_types_)
      {
          conf.names.push_back(joint_name + "/" + interface_type);
      }
  }

  for (const auto& interface_type : imu_interface_types_)
  {
      conf.names.push_back(imu_name_ + "/" + interface_type);
  }

  for (const auto& interface_type : foot_force_interface_types_)
  {
      conf.names.push_back(foot_force_name_ + "/" + interface_type);
  }

  return conf;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn RLQuadrupedController::on_init()
{
  try
  {
    auto node = get_node();

    policy_path_ = auto_declare<std::string>("policy_path", "");
    config_path_ = auto_declare<std::string>("config_path", "");

    cmd_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10, [this](const geometry_msgs::msg::Twist::SharedPtr msg)
      {
        cmd_[0] = msg->linear.x;
        cmd_[1] = msg->linear.y;
        cmd_[2] = msg->angular.z;
      });

    // string: sit, stand, move  
    mode_sub_ = node->create_subscription<std_msgs::msg::String>(
      "/mode", 10, [this](const std_msgs::msg::String::SharedPtr msg)
      {
        mode_ = msg->data;
      });

    joint_names_ = auto_declare<std::vector<std::string>>("joints", joint_names_);
    feet_names_ = auto_declare<std::vector<std::string>>("feet_names", feet_names_);
    command_interface_types_ =
        auto_declare<std::vector<std::string>>("command_interfaces", command_interface_types_);
    state_interface_types_ =
        auto_declare<std::vector<std::string>>("state_interfaces", state_interface_types_);

    command_prefix_ = auto_declare<std::string>("command_prefix", command_prefix_);
    base_name_ = auto_declare<std::string>("base_name", base_name_);

    // imu sensor
    imu_name_ = auto_declare<std::string>("imu_name", imu_name_);
    imu_interface_types_ = auto_declare<std::vector<std::string>>("imu_interfaces", state_interface_types_);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "on_init() failed: %s", e.what());
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
  }

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn RLQuadrupedController::on_configure(const rclcpp_lifecycle::State &)
{
  try
  {
    auto node = get_node();

    policy_ = torch::jit::load(policy_path_);
    YAML::Node config = YAML::LoadFile(config_path_);

    sit_angles_ = config["sit_angles"].as<std::vector<float>>();
    default_angles_ = config["default_angles"].as<std::vector<float>>();
    kps_ = config["kps"].as<std::vector<float>>();
    kds_ = config["kds"].as<std::vector<float>>();

    action_scale_ = config["action_scale"].as<std::vector<float>>();
    cmd_scale_ = config["cmd_scale"].as<std::vector<float>>();
    ang_vel_scale_ = config["ang_vel_scale"].as<float>();
    dof_pos_scale_ = config["dof_pos_scale"].as<float>();
    dof_vel_scale_ = config["dof_vel_scale"].as<float>();
    gravity_scale_ = config["gravity_scale"].as<float>();

    cmd_ = config["cmd_init"].as<std::vector<float>>();

    num_actions = config["num_actions"].as<int>();
    num_obs = config["num_obs"].as<int>();

    prev_action_ = torch::zeros({1, static_cast<long>(num_actions)});

    // Load low-pass filter cutoff frequency
    obs_lpf_fc_hz_ = 0.0;
    if (config["obs_lpf_cutoff_hz"]) {
      obs_lpf_fc_hz_ = config["obs_lpf_cutoff_hz"].as<double>();
    }

    // Initialize filter banks
    auto make_bank = [&](size_t n) {
      std::vector<LowPass1> v(n);
      for (auto &f : v) f.fc_hz = obs_lpf_fc_hz_;
      return v;
    };

    lpf_cmd_     = make_bank(3);
    lpf_ang_vel_ = make_bank(3);
    lpf_gravity_ = make_bank(3);
    lpf_pos_     = make_bank(12);
    lpf_vel_     = make_bank(12);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "on_configure() failed: %s", e.what());
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn RLQuadrupedController::on_activate(const rclcpp_lifecycle::State &)
{
  try
  { 
    // clear out vectors in case of restart
    ctrl_interfaces_.clear();

    // assign command interfaces
    for (auto& interface : command_interfaces_)
    {
        std::string interface_name = interface.get_interface_name();
        if (const size_t pos = interface_name.find('/'); pos != std::string::npos)
        {
            command_interface_map_[interface_name.substr(pos + 1)]->push_back(interface);
        }
        else
        {
            command_interface_map_[interface_name]->push_back(interface);
        }
    }

    // assign state interfaces
    for (auto& interface : state_interfaces_)
    {
        if (interface.get_prefix_name() == imu_name_)
        {
            ctrl_interfaces_.imu_state_interface_.emplace_back(interface);
        }
        else if (interface.get_prefix_name() == foot_force_name_)
        {
            ctrl_interfaces_.foot_force_state_interface_.emplace_back(interface);
        }
        else
        {
            state_interface_map_[interface.get_interface_name()]->push_back(interface);
        }
    }

    current_pos_ = get_current_pos();
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "on_activate() failed: %s", e.what());
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}
std::chrono::steady_clock::time_point last_time_;

controller_interface::return_type RLQuadrupedController::update(const rclcpp::Time &, const rclcpp::Duration &period)
{
  const double dt = period.seconds();
  // Check the frequency
  // auto now = std::chrono::steady_clock::now();
  // if (last_time_.time_since_epoch().count() != 0) {
  //   auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time_);
  //   double frequency = 1.0 / duration.count();  // 計算頻率（Hz）
  //   // std::cout << "Update frequency: " << frequency << " Hz" << std::endl;
  // }
  // last_time_ = now;

  if (mode_ != prev_mode_) {
    is_mode_change_ = true;
    step_ = 0;
    current_pos_ = get_current_pos();
    prev_mode_ = mode_;

    // Reset filters on mode change
    for (auto &f : lpf_cmd_) f.reset();
    for (auto &f : lpf_ang_vel_) f.reset();
    for (auto &f : lpf_gravity_) f.reset();
    for (auto &f : lpf_pos_) f.reset();
    for (auto &f : lpf_vel_) f.reset();
  } else {
    is_mode_change_ = false;
  }

  if (mode_ == "sit") {
    sit(step_, current_pos_);
  } else if (mode_ == "stand") {
    stand(step_, current_pos_);
  } else if (mode_ == "move") {
    move(dt);
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Invalid mode: %s", mode_.c_str());
  }

  if (step_ < steps_) {
    step_++;
  }

  return controller_interface::return_type::OK;
}

std::vector<float> RLQuadrupedController::get_current_pos()
{
  std::vector<float> pos(12);
  for (int i = 0; i < 12; ++i)
  {
    pos[i] = ctrl_interfaces_.joint_position_state_interface_[i].get().get_value();
  }
  return pos;
}

void RLQuadrupedController::sit(int step, std::vector<float> current_pos) 
{
  if (step < steps_) {
    double phase = float(step)/float(steps_);
    for (int i = 0; i < 12; ++i)
    {
      float target_pos = current_pos[i] * float(1 - phase) + sit_angles_[i] * phase;
      ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(target_pos);
      ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kps_[i]);
      ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kds_[i]);
    }
  } else {
    for (int i = 0; i < 12; ++i)
    {
      ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(sit_angles_[i]);
      ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kps_[i]);
      ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kds_[i]);
    }
  }
}

void RLQuadrupedController::stand(int step, std::vector<float> current_pos) 
{
  if (step < steps_) {
    double phase = float(step)/float(steps_);
    for (int i = 0; i < 12; ++i)
    {
      float target_pos = current_pos[i] * float(1 - phase) + default_angles_[i] * phase;
      ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(target_pos);
      ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kps_[i]);
      ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kds_[i]);
    }
  } else {
    for (int i = 0; i < 12; ++i)
    {
      ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(default_angles_[i]);
      ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kps_[i]);
      ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kds_[i]);
    }
  }
}

void RLQuadrupedController::move(double dt)
{
  std::vector<float> pos(12), vel(12), ang_vel(3), quat(4);

  for (int i = 0; i < 12; ++i)
  {
    pos[i] = ctrl_interfaces_.joint_position_state_interface_[i].get().get_value();
  }
  for (int i = 0; i < 12; ++i)
  {
    vel[i] = ctrl_interfaces_.joint_velocity_state_interface_[i].get().get_value();
  }

  quat[0] = ctrl_interfaces_.imu_state_interface_[0].get().get_value();
  quat[1] = ctrl_interfaces_.imu_state_interface_[1].get().get_value();
  quat[2] = ctrl_interfaces_.imu_state_interface_[2].get().get_value();
  quat[3] = ctrl_interfaces_.imu_state_interface_[3].get().get_value();
  ang_vel[0] = ctrl_interfaces_.imu_state_interface_[4].get().get_value();
  ang_vel[1] = ctrl_interfaces_.imu_state_interface_[5].get().get_value();
  ang_vel[2] = ctrl_interfaces_.imu_state_interface_[6].get().get_value();
  // ang_vel[0] = 0;
  // ang_vel[1] = 0;
  // ang_vel[2] = 0;

  std::vector<float> gravity(3);
  gravity[0] = 2 * (-quat[1] * quat[3] + quat[0] * quat[2]);
  gravity[1] = -2 * (quat[2] * quat[3] + quat[0] * quat[1]);
  gravity[2] = 1 - 2 * (quat[0] * quat[0] + quat[3] * quat[3]);

  // gravity[0] = 0;
  // gravity[1] = 0;
  // gravity[2] = -1;

  // Apply low-pass filters to sensor data
  for (int i = 0; i < 12; ++i) {
    pos[i] = lpf_pos_[i].update(pos[i], dt);
  }
  for (int i = 0; i < 12; ++i) {
    vel[i] = lpf_vel_[i].update(vel[i], dt);
  }
  for (int i = 0; i < 3; ++i) {
    ang_vel[i] = lpf_ang_vel_[i].update(ang_vel[i], dt);
  }
  for (int i = 0; i < 3; ++i) {
    gravity[i] = lpf_gravity_[i].update(gravity[i], dt);
  }

  torch::autograd::GradMode::set_enabled(false);

  std::vector<torch::Tensor> obs_vec = {
    torch::tensor(ang_vel) * ang_vel_scale_,
    torch::tensor(gravity),
    torch::tensor(latest_cmd_) * torch::tensor(cmd_scale_),
    (torch::tensor(pos) - torch::tensor(default_angles_)) * dof_pos_scale_,
    torch::tensor(vel) * dof_vel_scale_,
    prev_action_.squeeze()
  };

  auto obs = torch::cat(obs_vec).unsqueeze(0);

  // obs = obs.to(torch::kFloat32);

  obs = torch::clamp(obs, -100, 100);

  auto action = policy_.forward({obs}).toTensor().squeeze();
  prev_action_ = action;

  for (int i = 0; i < 12; ++i)
  {
    ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(action[i].item<float>() * action_scale_[i] + default_angles_[i]);
    ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kps_[i]);
    ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kds_[i]);
  }

  for (int i = 0; i < 3; ++i)
  {
    // latest_cmd_[i] = cmd_[i];
    latest_cmd_[i] = lpf_cmd_[i].update(cmd_[i], dt);
  }
}

} // namespace rl_quadruped_controller

PLUGINLIB_EXPORT_CLASS(rl_quadruped_controller::RLQuadrupedController, controller_interface::ControllerInterface)
