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
PLAYING_CARDS_0_TARGET_POS = [0.579195, 0.112946, 0.100693]
PLAYING_CARDS_0_TARGET_QUAT = [0.003043, 0.009320, -0.704666, 0.709472]  # (w, x, y, z)

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
                # Reach card
                checkers.reach("playing_cards_0", threshold=0.15),
                # Lift card
                (checkers.lift("playing_cards_0", default_height=0.03, threshold=0.03), [0]),
                # Place card at target position
                (checkers.pose_match(
                    "playing_cards_0",
                    target_pos=PLAYING_CARDS_0_TARGET_POS,
                    target_quat=PLAYING_CARDS_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.1,
                ), [1]),
            ]
        ),
    },
)


# Target poses for phone stand task
PHONE_0_TARGET_POS = [0.50, 0.20, 0.15]
PHONE_0_TARGET_QUAT = [1.0, 0.0, 0.0, 0.0]  # (w, x, y, z)

gym.register(
    id="DROID-PhoneStandKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "phone_stand_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                # Reach phone
                checkers.reach("phone_0", threshold=0.15),
                # Lift phone
                (checkers.lift("phone_0", default_height=0.05, threshold=0.03), [0]),
                # Place phone at target position (on phone_stand_0)
                (checkers.pose_match(
                    "phone_0",
                    target_pos=PHONE_0_TARGET_POS,
                    target_quat=PHONE_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.2,
                ), [1]),
            ]
        ),
    },
)


# Target poses for shoe kitchen task
SHOE_0_TARGET_POS = [0.500336, 0.190762, 0.074055]
SHOE_0_TARGET_QUAT = [-0.498205, -0.498078, 0.502018, 0.501684]  # (w, x, y, z)

gym.register(
    id="DROID-ShoeKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "shoe_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                # Reach shoe
                checkers.reach("shoe_0", threshold=0.15),
                # Lift shoe
                (checkers.lift("shoe_0", default_height=0.05, threshold=0.03), [0]),
                # Place shoe in shoe box
                (checkers.pose_match(
                    "shoe_0",
                    target_pos=SHOE_0_TARGET_POS,
                    target_quat=SHOE_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.3,
                ), [1]),
            ]
        ),
    },
)


# Target poses for rubiks box kitchen task
RUBIKSCUBE_0_TARGET_POS = [0.50, 0.262762, 0.149124]
RUBIKSCUBE_0_TARGET_QUAT = [0.008727, 0.0, 0.999962, 0.0]  # (w, x, y, z)

gym.register(
    id="DROID-RubiksBoxKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "rubiks_box_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                # Reach Rubiks cube
                checkers.reach("rubikscube_0", threshold=0.15),
                # Lift Rubiks cube
                (checkers.lift("rubikscube_0", default_height=0.05, threshold=0.03), [0]),
                # Place Rubiks cube in wooden box
                (checkers.pose_match(
                    "rubikscube_0",
                    target_pos=RUBIKSCUBE_0_TARGET_POS,
                    target_quat=RUBIKSCUBE_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.3,
                ), [1]),
            ]
        ),
    },
)


# Target poses for book kitchen task
BOOK_0_TARGET_POS = [0.37545, 0.294808, 0.179766]
BOOK_0_TARGET_QUAT = [1.0, 0.0, -0.000344, 0.000026]  # (w, x, y, z)

gym.register(
    id="DROID-BookKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "book_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                # Reach book
                checkers.reach("book_0", threshold=0.15),
                # Lift book
                (checkers.lift("book_0", default_height=0.05, threshold=0.03), [0]),
                # Place book on bookcase
                (checkers.pose_match(
                    "book_0",
                    target_pos=BOOK_0_TARGET_POS,
                    target_quat=BOOK_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.3,
                ), [1]),
            ]
        ),
    },
)


# Target poses for fork cup kitchen task
FORK_0_TARGET_POS = [0.50, 0.279115, 0.103304]
FORK_0_TARGET_QUAT = [0.707107, 0.707107, 0.0, 0.0]  # (w, x, y, z)

gym.register(
    id="DROID-ForkCupKitchen",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "fork_cup_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                # Reach fork
                checkers.reach("fork_0", threshold=0.15),
                # Lift fork
                (checkers.lift("fork_0", default_height=0.05, threshold=0.03), [0]),
                # Place fork in cup
                (checkers.pose_match(
                    "fork_0",
                    target_pos=FORK_0_TARGET_POS,
                    target_quat=FORK_0_TARGET_QUAT,
                    pos_threshold=0.05,
                    rot_threshold=0.3,
                ), [1]),
            ]
        ),
    },
)
