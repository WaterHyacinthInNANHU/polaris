"""
Trajectory recorder for Polaris DROID environments.

Records trajectories in HDF5 format for conversion to LeRobot dataset.
Adapted from SimEval DROID recorder with Polaris-specific observation structure.

Supports multi-env recording: each env_id tracks its own episode independently.
When an env reference is provided, camera images are rendered on demand via
Gaussian Splatting instead of reading from obs["splat"].
"""

import h5py
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PolarisTrajectoryRecorder:
    """Records trajectories from Polaris environment to HDF5 format.

    Records full DROID-format data including:
    - Joint positions and velocities (7D)
    - Gripper position and velocity (1D)
    - End-effector pose and velocity (7D pose, 6D velocity)
    - Camera images (external + wrist) at 720x1280 resolution
    - Action commands
    - Episode metadata with task_id and rubric results

    Supports per-env episode tracking for multi-env pipelines.
    When ``env`` is provided, camera images are rendered on demand via
    ``env.render_splat_for_env(env_id)`` instead of reading ``obs["splat"]``.
    """

    def __init__(self, output_dir: str, env: Any = None):
        """Initialize trajectory recorder.

        Args:
            output_dir: Directory to save trajectory file.
            env: Optional reference to ManagerBasedRLSplatEnv for on-demand
                 GS rendering. If provided, record_step() renders camera images
                 via env.render_splat_for_env(env_id).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"trajectory_{timestamp}.h5"

        # Environment reference for on-demand GS rendering
        self._env = env

        # Per-env episode state: env_id -> episode state dict
        self._per_env_state: dict[int, dict] = {}

        # Global episode counter (for unique HDF5 group names)
        self.episode_count = 0

    def set_env(self, env: Any):
        """Set the environment reference for on-demand GS rendering.

        Called after __init__ if the env was not available at construction time.

        Args:
            env: ManagerBasedRLSplatEnv instance with render_splat_for_env().
        """
        self._env = env

    def is_recording(self, env_id: int) -> bool:
        """Check if a specific env is currently recording.

        Args:
            env_id: Environment index.

        Returns:
            True if env_id has an active episode.
        """
        state = self._per_env_state.get(env_id)
        return state is not None and state["is_recording"]

    def start_episode(self, env_id: int, instruction: str, task_id: str):
        """Start recording a new episode for a specific env.

        Args:
            env_id: Environment index.
            instruction: Language instruction for the task.
            task_id: Task identifier (e.g., "DROID-BlockStackKitchen").
        """
        if self.is_recording(env_id):
            raise RuntimeError(
                f"Episode already in progress for env {env_id}. "
                "Call end_episode() or discard_episode() first."
            )

        episode_number = self.episode_count
        self.episode_count += 1

        self._per_env_state[env_id] = {
            "is_recording": True,
            "start_time": datetime.now().timestamp(),
            "episode_number": episode_number,
            "episode_data": {
                "observations": {
                    "external_cam": [],
                    "wrist_cam": [],
                    "joint_position": [],
                    "joint_velocity": [],
                    "gripper_position": [],
                    "gripper_velocity": [],
                    "ee_pose": [],
                    "ee_velocity": [],
                    "timestamp": [],
                },
                "actions": {
                    "joint_position_command": [],
                    "gripper_command": [],
                },
                "metadata": {
                    "instruction": instruction,
                    "task_id": task_id,
                    "env_id": env_id,
                    "fps": 15.0,
                },
            },
        }

        logger.info(
            "[Recorder] Started episode %d (env %d): '%s' (%s)",
            episode_number, env_id, instruction, task_id,
        )

    def record_step(self, env_id: int, obs: dict, action: np.ndarray):
        """Record a single timestep for a specific env.

        If ``self._env`` is set, camera images are rendered on demand via
        ``env.render_splat_for_env(env_id)``. Otherwise, images are read from
        ``obs["splat"]``.

        Args:
            env_id: Environment index.
            obs: Observation dictionary from environment.
                - obs["splat"]: Camera images (used when env is not set).
                - obs["policy"]: Proprioceptive state.
            action: Action array (8D: 7 joint commands + 1 gripper).
        """
        state = self._per_env_state.get(env_id)
        if state is None or not state["is_recording"]:
            raise RuntimeError(f"No episode in progress for env {env_id}.")

        episode_data = state["episode_data"]

        # --- Camera images: on-demand render or from obs ---
        if self._env is not None and hasattr(self._env, "render_splat_for_env"):
            splat_obs = self._env.render_splat_for_env(env_id)
        elif isinstance(obs, dict) and "splat" in obs:
            splat_obs = obs["splat"]
        else:
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            splat_obs = {"external_cam": placeholder, "wrist_cam": placeholder}

        external_cam = self._to_numpy(splat_obs.get("external_cam", np.zeros((720, 1280, 3), dtype=np.uint8)))
        wrist_cam = self._to_numpy(splat_obs.get("wrist_cam", np.zeros((720, 1280, 3), dtype=np.uint8)))

        # --- Proprioceptive state ---
        policy_obs = obs.get("policy", {})
        joint_pos = self._to_numpy(policy_obs.get("arm_joint_pos", np.zeros(7)))
        joint_vel = self._to_numpy(policy_obs.get("arm_joint_vel", np.zeros(7)))
        gripper_pos = self._to_numpy(policy_obs.get("gripper_pos", np.zeros(1)))
        gripper_vel = self._to_numpy(policy_obs.get("gripper_vel", np.zeros(1)))
        ee_pose = self._to_numpy(policy_obs.get("ee_pose", np.zeros(7)))
        ee_vel = self._to_numpy(policy_obs.get("ee_vel", np.zeros(6)))

        # Timestamp relative to episode start
        timestamp = datetime.now().timestamp() - state["start_time"]

        # Append observations
        episode_data["observations"]["external_cam"].append(external_cam)
        episode_data["observations"]["wrist_cam"].append(wrist_cam)
        episode_data["observations"]["joint_position"].append(joint_pos)
        episode_data["observations"]["joint_velocity"].append(joint_vel)
        episode_data["observations"]["gripper_position"].append(gripper_pos)
        episode_data["observations"]["gripper_velocity"].append(gripper_vel)
        episode_data["observations"]["ee_pose"].append(ee_pose)
        episode_data["observations"]["ee_velocity"].append(ee_vel)
        episode_data["observations"]["timestamp"].append(timestamp)

        # Append actions
        action = np.asarray(action, dtype=np.float32)
        joint_cmd = action[:7]
        gripper_cmd = action[7:8]
        episode_data["actions"]["joint_position_command"].append(joint_cmd)
        episode_data["actions"]["gripper_command"].append(gripper_cmd)

    def end_episode(self, env_id: int, rubric_result: Optional[dict] = None) -> bool:
        """End the current episode for env_id and save to HDF5.

        Args:
            env_id: Environment index.
            rubric_result: Optional rubric evaluation result with success/progress.

        Returns:
            True if the episode was saved successfully, False otherwise.
        """
        state = self._per_env_state.get(env_id)
        if state is None or not state["is_recording"]:
            logger.warning("[Recorder] No episode in progress for env %d.", env_id)
            return False

        episode_data = state["episode_data"]
        episode_number = state["episode_number"]
        episode_length = len(episode_data["observations"]["timestamp"])

        if episode_length == 0:
            logger.warning("[Recorder] Episode %d (env %d) has no data. Skipping.", episode_number, env_id)
            del self._per_env_state[env_id]
            return False

        logger.info("[Recorder] Ending episode %d (env %d, %d steps)", episode_number, env_id, episode_length)

        # Stack lists into arrays
        for obs_key in episode_data["observations"]:
            episode_data["observations"][obs_key] = np.stack(
                episode_data["observations"][obs_key], axis=0
            )
        for action_key in episode_data["actions"]:
            episode_data["actions"][action_key] = np.stack(
                episode_data["actions"][action_key], axis=0
            )

        # Metadata
        episode_data["metadata"]["episode_length"] = episode_length
        if rubric_result:
            episode_data["metadata"]["success"] = rubric_result.get("success", False)
            episode_data["metadata"]["progress"] = rubric_result.get("progress", 0.0)

        # Save to HDF5
        self._save_episode_to_hdf5(episode_number, episode_data)

        del self._per_env_state[env_id]
        return True

    def discard_episode(self, env_id: int):
        """Discard the current episode for env_id without saving.

        Args:
            env_id: Environment index.
        """
        if env_id in self._per_env_state:
            episode_number = self._per_env_state[env_id]["episode_number"]
            logger.info("[Recorder] Discarded episode %d (env %d)", episode_number, env_id)
            del self._per_env_state[env_id]

    def _save_episode_to_hdf5(self, episode_number: int, episode_data: dict):
        """Save episode data to HDF5 file.

        Args:
            episode_number: Global episode number for group naming.
            episode_data: Episode data dict with observations, actions, metadata.
        """
        with h5py.File(self.filepath, "a") as f:
            ep_group = f.create_group(f"episode_{episode_number}")

            # Save observations
            obs_group = ep_group.create_group("observations")
            for key, data in episode_data["observations"].items():
                obs_group.create_dataset(key, data=data, compression="gzip")

            # Save actions
            action_group = ep_group.create_group("actions")
            for key, data in episode_data["actions"].items():
                action_group.create_dataset(key, data=data, compression="gzip")

            # Save metadata
            meta_group = ep_group.create_group("metadata")
            for key, value in episode_data["metadata"].items():
                meta_group.attrs[key] = value

    def save(self):
        """Finalize: end any in-progress episodes and report."""
        for env_id in list(self._per_env_state.keys()):
            if self._per_env_state[env_id]["is_recording"]:
                logger.warning("[Recorder] Episode still in progress for env %d. Ending it.", env_id)
                self.end_episode(env_id)

        print(f"[Recorder] Saved {self.episode_count} episodes to {self.filepath}")

    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert torch tensor to numpy array, handling batch dimension."""
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
            if arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            return arr
        return np.asarray(tensor)
