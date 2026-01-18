import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import (
    ManagerBasedRLSplatEnv,
)
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import rubric system
from polaris.environments.rubrics import Rubric
from polaris.utils import DATA_PATH
import polaris.environments.rubrics.checkers as checkers


# =============================================================================
# Environment Registration
# =============================================================================

gym.register(
    id='DROID-BlockStackKitchen',
    entry_point=ManagerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "block_stack_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("green_cube", threshold=0.2),
                checkers.reach("wood_cube", threshold=0.2),
                (checkers.lift("green_cube", default_height=0.06, threshold=0.03), [0]),
                (checkers.lift("wood_cube", default_height=0.06, threshold=0.03), [1]),
                (checkers.is_within_xy("green_cube", "tray", 0.8), [2]),
                (checkers.is_within_xy("wood_cube", "tray", 0.8), [3]),
                (checkers.is_within_xy("green_cube", "wood_cube", 0.5), [4, 5]),
            ]
        ),
    },
    disable_env_checker=True,
    order_enforce=False,
)


gym.register(
    id="DROID-FoodBussing",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "food_bussing/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("ice_cream_", threshold=0.2),
                checkers.reach("grapes", threshold=0.2),
                (checkers.lift("ice_cream_", threshold=0.06), [0]),
                (checkers.lift("grapes", threshold=0.06), [1]),
                (
                    checkers.is_within_xy("ice_cream_", "bowl", percent_threshold=0.8),
                    [2],
                ),
                (checkers.is_within_xy("grapes", "bowl", percent_threshold=0.8), [3]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PanClean",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "pan_clean/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("sponge", threshold=0.2),
                (checkers.lift("sponge", threshold=0.09, default_height=0.0), [0]),
                (checkers.is_within_xy("sponge", "pan", percent_threshold=0.8), [1]),
            ]
        ),
    },
)


gym.register(
    id="DROID-MoveLatteCup",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "move_latte_cup/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("latteartcup_eval", threshold=0.2),
                (checkers.lift("latteartcup_eval", threshold=0.04), [0]),
                (checkers.is_within_xy("latteartcup_eval", "cuttingboard_eval", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-OrganizeTools",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "organize_tools/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("scissor", threshold=0.2),
                (checkers.lift("scissor", threshold=0.04), [0]),
                (checkers.is_within_xy("scissor", "container_01", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-TapeIntoContainer",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "tape_into_container/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("tape_00", threshold=0.2),
                (checkers.lift("tape_00", threshold=0.04), [0]),
                (checkers.is_within_xy("tape_00", "container_02", percent_threshold=0.8), [1]),
            ]
        ),
    },
)


# Target position for Rubik's cube placement task
RUBIKS_CUBE_TARGET_POS = [0.50, 0.10, 0.08]

gym.register(
    id="DROID-RubiksCubeKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "rubiks_cube_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("rubiks_cube", threshold=0.15),
                (checkers.lift("rubiks_cube", default_height=0.08, threshold=0.03), [0]),
                (checkers.pose_match(
                    "rubiks_cube",
                    target_pos=RUBIKS_CUBE_TARGET_POS,
                    target_quat=None,
                    pos_threshold=0.05,
                    rot_threshold=0.1,
                ), [1]),
            ]
        ),
    },
)


# Target poses for playing cards placement task
PLAYING_CARDS_0_TARGET_POS = [0.432594, 0.077528, 0.058701]
PLAYING_CARDS_0_TARGET_QUAT = [-0.030789, 0.006440, -0.700888, 0.712577]  # (w, x, y, z)
PLAYING_CARDS_1_TARGET_POS = [0.417737, 0.270094, 0.059539]
PLAYING_CARDS_1_TARGET_QUAT = [-0.687985, 0.725225, 0.013425, -0.023347]  # (w, x, y, z)

gym.register(
    id="DROID-PlayingCardsKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "playing_cards_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                # Reach each card
                checkers.reach("playing_cards_0", threshold=0.15),
                checkers.reach("playing_cards_1", threshold=0.15),
                # Lift each card
                (checkers.lift("playing_cards_0", default_height=0.03, threshold=0.03), [0]),
                (checkers.lift("playing_cards_1", default_height=0.03, threshold=0.03), [1]),
                # Place each card at target position
                (checkers.pose_match(
                    "playing_cards_0",
                    target_pos=PLAYING_CARDS_0_TARGET_POS,
                    target_quat=PLAYING_CARDS_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.1,
                ), [2]),
                (checkers.pose_match(
                    "playing_cards_1",
                    target_pos=PLAYING_CARDS_1_TARGET_POS,
                    target_quat=PLAYING_CARDS_1_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.1,
                ), [3]),
            ]
        ),
    },
)
