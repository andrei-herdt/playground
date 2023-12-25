import mujoco
from mujoco import mjx

import jax
from jax import numpy as jp

from typing import Any, Dict, Tuple, Union

from brax.envs.base import Env, State
from brax.base import Base, Motion, Transform
from brax import math

from etils import epath

from ml_collections import config_dict

import numpy as np

from flax import struct


class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
    ):
        """Initializes MjxEnv.

        Args:
          mj_model: mujoco.MjModel
          physics_steps_per_control_step: the number of times to step the physics
            pipeline for each environment step
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return (
                mjx.step(self.sys, data),
                None,
            )

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def backend(self) -> str:
        return "mjx"

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = (
            data.xpos[1:, :]
            - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        )
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd


# Quadruped
def get_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.5,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-2.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0,
                        # L2 regularization of joint torques, |tau|^2.
                        # torques=-0.0002,
                        torques=-0.00001,

                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=2,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.1,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config


class BarkourEnv(MjxEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        **kwargs,
    ):
        path = epath.Path(epath.resource_path("mujoco")) / (
            "mjx/benchmark/model/barkour_v0/assets"
        )
        mj_model = mujoco.MjModel.from_xml_path(
            (path / "barkour_v0_mjx.xml").as_posix()
        )
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 4
        mj_model.opt.ls_iterations = 6

        physics_steps_per_control_step = 10
        kwargs["physics_steps_per_control_step"] = kwargs.get(
            "physics_steps_per_control_step", physics_steps_per_control_step
        )
        super().__init__(mj_model=mj_model, **kwargs)

        self.torso_idx = 1
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._reset_horizon = 500
        self._feet_index = jp.array([3, 6, 9, 12])
        # local positions for each foot
        self._feet_pos = jp.array(
            [
                [-0.191284, -0.0191638, 0.013],
                [-0.191284, -0.0191638, -0.013],
                [-0.191284, -0.0191638, 0.013],
                [-0.191284, -0.0191638, -0.013],
            ]
        )
        self._init_q = mj_model.keyframe("standing").qpos
        self._default_ap_pose = mj_model.keyframe("standing").qpos[7:]
        self.reward_config = get_config()
        self.lowers = self._default_ap_pose - jp.array([0.2, 0.8, 0.8] * 4)
        self.uppers = self._default_ap_pose + jp.array([0.2, 0.8, 0.8] * 4)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.6, 1.0]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        qpos = jp.array(self._init_q)
        qvel = jp.zeros(self.model.nv)
        new_cmd = self.sample_command(key)
        data = self.pipeline_init(qpos, qvel)

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_vel": jp.zeros(12),
            "last_contact_buffer": jp.zeros((20, 4), dtype=bool),
            "command": new_cmd,
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "obs_history": jp.zeros(15 * 31),
            "reward_tuple": {
                "tracking_lin_vel": 0.0,
                "tracking_ang_vel": 0.0,
                "lin_vel_z": 0.0,
                "ang_vel_xy": 0.0,
                "orientation": 0.0,
                "torque": 0.0,
                "action_rate": 0.0,
                "stand_still": 0.0,
                "feet_air_time": 0.0,
                "foot_slip": 0.0,
            },
            "step": 0,
        }

        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, x, xd, state_info)
        reward, done = jp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]
        state = State(data, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, rng_noise, cmd_rng = jax.random.split(state.info["rng"], 3)

        # physics step
        cur_action = jp.array(action)
        action = action[:12] * self._action_scale
        motor_targets = jp.clip(
            action + self._default_ap_pose, self.lowers, self.uppers
        )
        data = self.pipeline_step(state.pipeline_state, motor_targets)

        # observation data
        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, x, xd, state.info)
        obs_noise = self._obs_noise * jax.random.uniform(
            rng_noise, obs.shape, minval=-1, maxval=1
        )
        qpos, qvel = data.qpos, data.qvel
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]

        # foot contact data based on z-position
        foot_contact = 0.017 - self._get_feet_pos_vel(x, xd)[0][:, 2]
        contact = foot_contact > -1e-3  # a mm or less off the floor
        contact_filt_mm = jp.logical_or(contact, state.info["last_contact"])
        contact_filt_cm = jp.logical_or(
            foot_contact > -1e-2, state.info["last_contact"]
        )
        first_contact = (state.info["feet_air_time"] > 0) * (contact_filt_mm)
        state.info["feet_air_time"] += self.dt

        # reward
        reward_tuple = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd)
                * self.reward_config.rewards.scales.tracking_lin_vel
            ),
            "tracking_ang_vel": (
                self._reward_tracking_ang_vel(state.info["command"], x, xd)
                * self.reward_config.rewards.scales.tracking_ang_vel
            ),
            "lin_vel_z": (
                self._reward_lin_vel_z(xd) * self.reward_config.rewards.scales.lin_vel_z
            ),
            "ang_vel_xy": (
                self._reward_ang_vel_xy(xd)
                * self.reward_config.rewards.scales.ang_vel_xy
            ),
            "orientation": (
                self._reward_orientation(x)
                * self.reward_config.rewards.scales.orientation
            ),
            "torque": (
                self._reward_torques(data.qfrc_actuator)
                * self.reward_config.rewards.scales.torques
            ),
            "action_rate": (
                self._reward_action_rate(cur_action, state.info["last_act"])
                * self.reward_config.rewards.scales.action_rate
            ),
            "stand_still": (
                self._reward_stand_still(
                    state.info["command"], joint_angles, self._default_ap_pose
                )
                * self.reward_config.rewards.scales.stand_still
            ),
            "feet_air_time": (
                self._reward_feet_air_time(
                    state.info["feet_air_time"],
                    first_contact,
                    state.info["command"],
                )
                * self.reward_config.rewards.scales.feet_air_time
            ),
            "foot_slip": (
                self._reward_foot_slip(x, xd, contact_filt_cm)
                * self.reward_config.rewards.scales.foot_slip
            ),
        }
        reward = sum(reward_tuple.values())
        reward = jp.clip(reward * self.dt, 0.0, 10000.0)

        # state management
        state.info["last_act"] = cur_action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["last_contact_buffer"] = jp.roll(
            state.info["last_contact_buffer"], 1, axis=0
        )
        state.info["last_contact_buffer"] = (
            state.info["last_contact_buffer"].at[0].set(contact)
        )
        state.info["reward_tuple"] = reward_tuple
        state.info["step"] += 1
        state.info.update(rng=rng)

        # resetting logic if joint limits are reached or robot is falling
        done = 0.0
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)
        done = jp.where(
            jp.logical_or(
                jp.any(joint_angles < 0.98 * self.lowers),
                jp.any(joint_angles > 0.98 * self.uppers),
            ),
            1.0,
            done,
        )
        done = jp.where(x.pos[self.torso_idx, 2] < 0.18, 1.0, done)

        # termination reward
        reward += jp.where(
            (done == 1.0) & (state.info["step"] < self._reset_horizon),
            self.reward_config.rewards.scales.termination,
            0.0,
        )

        # when done, sample new command if more than _reset_horizon timesteps
        # achieved
        state.info["command"] = jp.where(
            (done == 1.0) & (state.info["step"] > self._reset_horizon),
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            (done == 1.0) | (state.info["step"] > self._reset_horizon),
            0,
            state.info["step"],
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self.torso_idx])[1]
        for k in state.info["reward_tuple"].keys():
            state.metrics[k] = state.info["reward_tuple"][k]

        state = state.replace(
            pipeline_state=data, obs=obs + obs_noise, reward=reward, done=done
        )
        return state

    def _get_obs(
        self, qpos: jax.Array, x: Transform, xd: Motion, state_info: Dict[str, Any]
    ) -> jax.Array:
        # Get observations:
        # yaw_rate,  projected_gravity, command,  motor_angles, last_action

        inv_base_orientation = math.quat_inv(x.rot[0])
        local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)
        cmd = state_info["command"]

        obs_list = []
        # yaw rate
        obs_list.append(jp.array([local_rpyrate[2]]) * 0.25)
        # projected gravity
        obs_list.append(math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation))
        # command
        obs_list.append(cmd * jp.array([2.0, 2.0, 0.25]))
        # motor angles
        angles = qpos[7:19]
        obs_list.append(angles - self._default_ap_pose)
        # last action
        obs_list.append(state_info["last_act"])

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

        # stack observations through time
        single_obs_size = len(obs)
        state_info["obs_history"] = jp.roll(state_info["obs_history"], single_obs_size)
        state_info["obs_history"] = (
            jp.array(state_info["obs_history"]).at[:single_obs_size].set(obs)
        )
        return state_info["obs_history"]

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self, commands: jax.Array, joint_angles: jax.Array, default_angles: jax.Array
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - default_angles)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _get_feet_pos_vel(
        self, x: Transform, xd: Motion
    ) -> Tuple[jax.Array, jax.Array]:
        offset = Transform.create(pos=self._feet_pos)
        pos = x.take(self._feet_index).vmap().do(offset).pos
        vel = offset.vmap().do(xd.take(self._feet_index)).vel
        return pos, vel

    def _reward_foot_slip(
        self, x: Transform, xd: Motion, contact_filt: jax.Array
    ) -> jax.Array:
        # Get feet velocities
        _, foot_world_vel = self._get_feet_pos_vel(x, xd)
        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_world_vel[:, :2]) * contact_filt.reshape((-1, 1)))


def domain_randomize(sys, rng):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-10, -5)
        param = (
            jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1])
            + sys.actuator_gainprm[:, 0]
        )
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


@struct.dataclass
class State(Base):
    """Environment state for training and inference with brax.

    Args:
      pipeline_state: the physics state, mjx.Data
      obs: environment observations
      reward: environment reward
      done: boolean, True if the current episode has terminated
      metrics: metrics that get tracked per environment step
      info: environment variables defined and updated by the environment reset
        and step functions
    """

    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)
