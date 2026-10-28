# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import LizardRoughEnvCfg

import math

@configclass


class LizardFlatEnvCfg(LizardRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Friction
        #self.events.physics_material.params["static_friction_range"]: (0.4, 0.6)
        #self.events.physics_material.params["dynamic_friction_range"]: (0.2, 0.4)
        #self.scene.terrain.physics_material.static_friction= 0.8
        #self.scene.terrain.physics_material.dynamic_friction= 0.6

        self.events.reset_base.params = {
            "pose_range": {
               "x":   (0.00, 0.0),
                "y":   (0.0, 0.0),
                "yaw": (0.0,  0.0), 
            },
            "velocity_range": {
                "x":    (-0.05, 0.05),
                "y":    (-0.05, 0.05),
                "z":    (-0.05, 0.05),
                "roll": (0.0,  0.0),
                "pitch":(0.0,  0.0),
                "yaw":  (00.0,  0.0),
           },
        }

        # override body rewards
        self.events.add_base_mass.params["asset_cfg"].body_names=["Front", "Mid", "Back", "Tail"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.base_com.params["asset_cfg"].body_names="Mid"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "Mid"

        self.rewards.feet_air_time.params["sensor_cfg"].body_names=["FR","FL","HR","HL"]
        self.rewards.feet_air_time.weight = 0.04
        self.rewards.feet_air_time.params["threshold"] = 0.5

        self.rewards.undesired_contacts.params["sensor_cfg"].body_names=["Front", "Mid", "Back","Tail"]
        self.rewards.undesired_contacts.weight=0.0

        self.terminations.base_contact.params["sensor_cfg"].body_names=["Front", "Mid", "Back"]
        self.terminations.base_contact.params["threshold"] = 20.0

        #self.rewards.joint_deviation.weight = -0.1

        self.commands.base_velocity.ranges.lin_vel_x=(0,0.4) # only want to go forward
        self.commands.base_velocity.ranges.lin_vel_y=(0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z=(0.0,0.0) 

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)*0.25
        
        self.rewards.lin_vel_z_l2.weight = -0.1

        #self.rewards.dof_torques_l2.weight = -5.0e-5
        
        self.rewards.dof_pos_limits.weight = -1.0
        #self.rewards.undesired_contacts.weight = -0.5
        #self.rewards.track_ang_vel_z_exp.weight = 0.5
        #self.rewards.ang_vel_xy_l2.weight = -0.05
        #self.rewards.flat_orientation_l2.weight = 0.0
        #self.rewards.action_rate_l2.weight= -0.01

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel = None
        self.observations.policy.projected_gravity = None
        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None

        self.observations.policy.actions.history_length = 8
        
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class LizardFlatEnvCfg_PLAY(LizardFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)  # choose video speed
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # no resampling during an episode (episodes are 1000 long)
        self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        # avoid random “standing still” envs
        self.commands.base_velocity.rel_standing_envs = 0.0

        # camera settings
        self.viewer.eye = (0.5, -0.6, 0.3)       # in front, a bit to the right, a bit above
        self.viewer.lookat = (0.0, 0.0, 0.1)     # near the robot’s mid-body
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"