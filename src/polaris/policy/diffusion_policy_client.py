"""
Polaris InferenceClient for Diffusion Policy.

This client connects to a diffusion policy WebSocket server and provides
the InferenceClient interface expected by Polaris.

Usage:
    # Start the diffusion policy server first:
    python my_regrasp/training/serve_diffusion_policy.py --checkpoint <path> --port 8000

    # Then run evaluation with this client:
    uv run scripts/eval.py --environment DROID-FoodBussing \
        --policy.client DiffusionPolicy --policy.port 8000
"""

import numpy as np
import cv2
from openpi_client import websocket_client_policy, image_tools
from polaris.policy.abstract_client import InferenceClient
from polaris.config import PolicyArgs


@InferenceClient.register(client_name="DiffusionPolicy")
class DiffusionPolicyClient(InferenceClient):
    """
    Polaris client for diffusion policy inference.

    This client:
    1. Connects to a diffusion policy WebSocket server
    2. Preprocesses observations from Polaris format
    3. Sends observations to the server
    4. Returns actions in the expected format

    The client supports action chunking - it receives a sequence of actions
    from the server and executes them in open-loop before requesting new actions.
    """

    def __init__(self, args: PolicyArgs) -> None:
        """
        Initialize the diffusion policy client.

        Args:
            args: Policy arguments containing host, port, and other settings
        """
        self.args = args

        # Get open_loop_horizon from args, default to 8 (typical for diffusion policy)
        if args.open_loop_horizon is None:
            self.open_loop_horizon = 8
        else:
            self.open_loop_horizon = args.open_loop_horizon

        # Image size - should match what the server expects
        self.image_size = getattr(args, 'image_size', 224)

        # Connect to server
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )

        # Get server metadata
        metadata = self.client.get_server_metadata()
        if metadata:
            print(f"Connected to diffusion policy server:")
            print(f"  n_obs_steps: {metadata.get('n_obs_steps', 'N/A')}")
            print(f"  n_action_steps: {metadata.get('n_action_steps', 'N/A')}")
            print(f"  horizon: {metadata.get('horizon', 'N/A')}")

            # Use server's n_action_steps if available
            if 'n_action_steps' in metadata:
                self.open_loop_horizon = min(
                    self.open_loop_horizon,
                    metadata['n_action_steps']
                )

        # Action chunking state
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    @property
    def rerender(self) -> bool:
        """
        Whether to rerender the visualization.

        Returns True when we need new observations (at the start or after
        completing the current action chunk).
        """
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        """Reset client state for new episode."""
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

        # Notify server of reset
        try:
            self.client.infer({"reset": True})
        except Exception:
            pass  # Server might not support reset signal

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the diffusion policy.

        Args:
            obs: Observation dict from Polaris with keys:
                - "splat": {"external_cam": np.ndarray, "wrist_cam": np.ndarray}
                - "policy": {"arm_joint_pos": torch.Tensor, "gripper_pos": torch.Tensor}
            instruction: Language instruction (currently unused by diffusion policy)
            return_viz: Whether to return visualization

        Returns:
            Tuple of (action, visualization):
                - action: np.ndarray (8,) - 7 joint positions + 1 gripper
                - visualization: np.ndarray or None - concatenated camera views
        """
        viz = None

        # Check if we need to query the server for new actions
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            # Extract and preprocess observation
            curr_obs = self._extract_observation(obs)

            # Reset chunk counter
            self.actions_from_chunk_completed = 0

            # Resize images to expected size
            external_image = image_tools.resize_with_pad(
                curr_obs["external_image"], self.image_size, self.image_size
            )
            wrist_image = image_tools.resize_with_pad(
                curr_obs["wrist_image"], self.image_size, self.image_size
            )

            # Build request in OpenPI format
            request_data = {
                "observation/exterior_image_1_left": external_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }

            # Query server
            server_response = self.client.infer(request_data)
            self.pred_action_chunk = server_response["actions"]

            # Create visualization
            viz = np.concatenate([external_image, wrist_image], axis=1)

        # Generate visualization if requested but not already created
        if return_viz and viz is None:
            curr_obs = self._extract_observation(obs)
            external_image = image_tools.resize_with_pad(
                curr_obs["external_image"], self.image_size, self.image_size
            )
            wrist_image = image_tools.resize_with_pad(
                curr_obs["wrist_image"], self.image_size, self.image_size
            )
            viz = np.concatenate([external_image, wrist_image], axis=1)

        # Get current action from chunk
        if self.pred_action_chunk is None:
            raise ValueError("No action chunk available. Call infer() first.")

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # Binarize gripper action (standard for DROID)
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        return action.astype(np.float32), viz

    def _extract_observation(self, obs_dict: dict) -> dict:
        """
        Extract observations from Polaris format.

        Args:
            obs_dict: Raw observation from Polaris environment

        Returns:
            Processed observation dict with:
                - external_image: np.ndarray (H, W, 3)
                - wrist_image: np.ndarray (H, W, 3)
                - joint_position: np.ndarray (7,)
                - gripper_position: np.ndarray (1,)
        """
        # Extract images from splat rendering
        external_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        # Extract proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        # Handle batch dimension
        if joint_position.ndim > 1:
            joint_position = joint_position[0]
        if gripper_position.ndim > 1:
            gripper_position = gripper_position[0]

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def visualize(self, obs: dict) -> np.ndarray:
        """
        Create visualization of current observation.

        Args:
            obs: Observation dict from Polaris

        Returns:
            Concatenated camera views as np.ndarray
        """
        curr_obs = self._extract_observation(obs)
        external_image = image_tools.resize_with_pad(
            curr_obs["external_image"], self.image_size, self.image_size
        )
        wrist_image = image_tools.resize_with_pad(
            curr_obs["wrist_image"], self.image_size, self.image_size
        )
        return np.concatenate([external_image, wrist_image], axis=1)
