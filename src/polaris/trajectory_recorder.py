"""
Trajectory recorder for Polaris DROID environments.

Records trajectories in HDF5 format for conversion to LeRobot dataset.
Adapted from SimEval DROID recorder with Polaris-specific observation structure.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional


class PolarisTrajectoryRecorder:
    """Records trajectories from Polaris environment to HDF5 format.

    Records full DROID-format data including:
    - Joint positions and velocities (7D)
    - Gripper position and velocity (1D)
    - End-effector pose and velocity (7D pose, 6D velocity)
    - Camera images (external + wrist) at 720x1280 resolution
    - Action commands
    - Episode metadata with task_id and rubric results

    Key differences from SimEval recorder:
    - Camera images come from obs["splat"] group (not obs["policy"])
    - Proprioceptive state comes from obs["policy"] group
    - Uses original 720x1280 resolution (no downsampling)
    - Includes task_id and rubric success/progress metadata
    """

    def __init__(self, output_dir: str):
        """Initialize trajectory recorder.

        Args:
            output_dir: Directory to save trajectory file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"trajectory_{timestamp}.h5"

        # Episode tracking
        self.current_episode = None
        self.episode_count = 0
        self.episode_data = {}

        # Recording state
        self.is_recording = False
        self.start_time = None

    def start_episode(self, instruction: str, task_id: str):
        """Start recording a new episode.

        Args:
            instruction: Language instruction for the task
            task_id: Task identifier (e.g., "DROID-BlockStackKitchen")
        """
        if self.is_recording:
            raise RuntimeError("Episode already in progress. Call end_episode() first.")

        self.is_recording = True
        self.start_time = datetime.now().timestamp()
        self.current_episode = self.episode_count

        # Initialize episode data storage
        self.episode_data = {
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
                "fps": 15.0,  # Polaris runs at 15Hz (dt=1/120, decimation=8)
            }
        }

        print(f"[Recorder] Started episode {self.episode_count}: '{instruction}' ({task_id})")

    def record_step(self, obs: dict, action: np.ndarray):
        """Record a single timestep.

        Args:
            obs: Observation dictionary from environment
                - obs["splat"]: Camera images (external_cam, wrist_cam)
                - obs["policy"]: Proprioceptive state
            action: Action array (8D: 7 joint commands + 1 gripper)
        """
        if not self.is_recording:
            raise RuntimeError("No episode in progress. Call start_episode() first.")

        # Extract camera images from splat group (720x1280 RGB)
        splat_obs = obs["splat"]
        external_cam = self._to_numpy(splat_obs["external_cam"])  # (720, 1280, 3) uint8
        wrist_cam = self._to_numpy(splat_obs["wrist_cam"])  # (720, 1280, 3) uint8

        # Extract proprioceptive state from policy group
        policy_obs = obs["policy"]
        joint_pos = self._to_numpy(policy_obs["arm_joint_pos"])  # (7,)
        joint_vel = self._to_numpy(policy_obs["arm_joint_vel"])  # (7,)
        gripper_pos = self._to_numpy(policy_obs["gripper_pos"])  # (1,)
        gripper_vel = self._to_numpy(policy_obs["gripper_vel"])  # (1,)

        # Extract end-effector state
        ee_pose = self._to_numpy(policy_obs["ee_pose"])  # (7,) xyz + quat
        ee_vel = self._to_numpy(policy_obs["ee_vel"])  # (6,) linear + angular

        # Record timestamp (relative to episode start)
        timestamp = datetime.now().timestamp() - self.start_time

        # Append to episode data
        self.episode_data["observations"]["external_cam"].append(external_cam)
        self.episode_data["observations"]["wrist_cam"].append(wrist_cam)
        self.episode_data["observations"]["joint_position"].append(joint_pos)
        self.episode_data["observations"]["joint_velocity"].append(joint_vel)
        self.episode_data["observations"]["gripper_position"].append(gripper_pos)
        self.episode_data["observations"]["gripper_velocity"].append(gripper_vel)
        self.episode_data["observations"]["ee_pose"].append(ee_pose)
        self.episode_data["observations"]["ee_velocity"].append(ee_vel)
        self.episode_data["observations"]["timestamp"].append(timestamp)

        # Record actions
        action = np.asarray(action, dtype=np.float32)
        joint_cmd = action[:7]  # First 7 are joint commands
        gripper_cmd = action[7:8]  # Last one is gripper

        self.episode_data["actions"]["joint_position_command"].append(joint_cmd)
        self.episode_data["actions"]["gripper_command"].append(gripper_cmd)

    def end_episode(self, rubric_result: Optional[dict] = None):
        """End the current episode and save to HDF5.

        Args:
            rubric_result: Optional rubric evaluation result with success/progress
        """
        if not self.is_recording:
            print("[Recorder] Warning: No episode in progress.")
            return

        # Stack all timesteps into arrays
        episode_length = len(self.episode_data["observations"]["timestamp"])

        if episode_length == 0:
            print("[Recorder] Warning: Episode has no data. Skipping.")
            self.is_recording = False
            return

        print(f"[Recorder] Ending episode {self.episode_count} ({episode_length} steps)")

        # Convert lists to numpy arrays
        for obs_key in self.episode_data["observations"]:
            self.episode_data["observations"][obs_key] = np.stack(
                self.episode_data["observations"][obs_key], axis=0
            )

        for action_key in self.episode_data["actions"]:
            self.episode_data["actions"][action_key] = np.stack(
                self.episode_data["actions"][action_key], axis=0
            )

        # Add episode length and rubric result to metadata
        self.episode_data["metadata"]["episode_length"] = episode_length
        if rubric_result:
            self.episode_data["metadata"]["success"] = rubric_result.get("success", False)
            self.episode_data["metadata"]["progress"] = rubric_result.get("progress", 0.0)

        # Save to HDF5
        self._save_episode_to_hdf5()

        # Increment episode counter and reset state
        self.episode_count += 1
        self.is_recording = False
        self.episode_data = {}

    def _save_episode_to_hdf5(self):
        """Save current episode data to HDF5 file."""
        with h5py.File(self.filepath, "a") as f:
            ep_group = f.create_group(f"episode_{self.current_episode}")

            # Save observations
            obs_group = ep_group.create_group("observations")
            for key, data in self.episode_data["observations"].items():
                obs_group.create_dataset(key, data=data, compression="gzip")

            # Save actions
            action_group = ep_group.create_group("actions")
            for key, data in self.episode_data["actions"].items():
                action_group.create_dataset(key, data=data, compression="gzip")

            # Save metadata
            meta_group = ep_group.create_group("metadata")
            for key, value in self.episode_data["metadata"].items():
                if isinstance(value, str):
                    meta_group.attrs[key] = value
                else:
                    meta_group.attrs[key] = value

    def save(self):
        """Finalize and close the HDF5 file."""
        if self.is_recording:
            print("[Recorder] Warning: Episode still in progress. Ending it.")
            self.end_episode()

        print(f"[Recorder] Saved {self.episode_count} episodes to {self.filepath}")

    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy array, handling batch dimension."""
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
            # Remove batch dimension if present
            if arr.shape[0] == 1:
                arr = arr[0]
            return arr
        return np.asarray(tensor)
