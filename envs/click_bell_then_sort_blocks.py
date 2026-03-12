from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class click_bell_then_sort_blocks(Base_Task):
    """
    Stage 1: Click a bell (from click_bell).
    Stage 2: Place mouse on pad (from place_mouse_pad).
    Simple single-arm tasks on opposite sides.
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1: Bell (from click_bell) ---
        bell_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while abs(bell_pose.p[0]) < 0.05:
            bell_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, -0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=bell_pose,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = (
            self.is_left_gripper_close if self.bell.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )

        # --- Stage 2: Mouse + pad (from place_mouse_pad) on opposite side ---
        bell_side = 1 if bell_pose.p[0] > 0 else -1
        mouse_xlim = [-0.25, -0.05] if bell_side > 0 else [0.05, 0.25]

        mouse_pose = rand_pose(
            xlim=mouse_xlim,
            ylim=[0.0, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        self.mouse_id = np.random.choice([0, 1, 2], 1)[0]
        self.mouse = create_actor(
            scene=self,
            pose=mouse_pose,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
        )
        self.mouse.set_mass(0.05)

        pad_pose = rand_pose(
            xlim=mouse_xlim,
            ylim=[0.0, 0.15],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        while np.sqrt((pad_pose.p[0] - mouse_pose.p[0])**2
                       + (pad_pose.p[1] - mouse_pose.p[1])**2) < 0.1:
            pad_pose = rand_pose(
                xlim=mouse_xlim,
                ylim=[0.0, 0.15],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )

        colors = {"Red": (1, 0, 0), "Green": (0, 1, 0), "Blue": (0, 0, 1)}
        color_items = list(colors.items())
        self.color_name, self.color_value = color_items[np.random.choice(len(color_items))]
        self.pad = create_box(
            scene=self,
            pose=pad_pose,
            half_size=[0.035, 0.065, 0.0005],
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.pad, padding=0.12)
        self.add_prohibit_area(self.mouse, padding=0.03)
        self.pad_target_pose = self.pad.get_pose().p.tolist() + [0, 0, 0, 1]

    def play_once(self):
        # === Stage 1: Click the bell ===
        bell_arm = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(
            self.bell, arm_tag=bell_arm, pre_grasp_dis=0.1, grasp_dis=0.1,
            contact_point_id=0,
        ))
        self.move(self.move_by_displacement(bell_arm, z=-0.045))
        self.move(self.move_by_displacement(bell_arm, z=0.045))

        # Return arm before mouse task
        self.move(self.back_to_origin(bell_arm))

        # === Stage 2: Place mouse on pad ===
        mouse_arm = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.mouse, arm_tag=mouse_arm, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=mouse_arm, z=0.1))
        self.move(
            self.place_actor(
                self.mouse, arm_tag=mouse_arm,
                target_pose=self.pad_target_pose,
                constrain="align", pre_dis=0.07, dis=0.005,
            ))

        self.info["info"] = {
            "{A}": f"050_bell/base{self.bell_id}",
            "{B}": f"047_mouse/base{self.mouse_id}",
            "{C}": f"{self.color_name}",
        }
        return self.info

    def check_success(self):
        mouse_pose = self.mouse.get_pose().p
        target_pos = self.pad.get_pose().p
        eps = 0.015
        return (np.all(abs(mouse_pose[:2] - target_pos[:2]) < eps)
                and self.is_left_gripper_open()
                and self.is_right_gripper_open())
