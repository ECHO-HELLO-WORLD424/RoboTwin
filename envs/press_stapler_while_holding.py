from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class press_stapler_while_holding(Base_Task):
    """
    Stage 1: Place mouse on mouse pad.
    Stage 2: Press stapler (with other arm or same arm after placing).
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1: Mouse + Pad (from place_mouse_pad) ---
        mouse_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        while abs(mouse_pos.p[0]) < 0.12:
            mouse_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
        self.mouse_id = np.random.choice([0, 1, 2], 1)[0]
        self.mouse = create_actor(
            scene=self,
            pose=mouse_pos,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
        )
        self.mouse.set_mass(0.05)

        # Pad on the same side as mouse
        pad_xlim = [0.05, 0.25] if mouse_pos.p[0] > 0 else [-0.25, -0.05]
        pad_pose = rand_pose(
            xlim=pad_xlim,
            ylim=[0.05, 0.2],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        while (np.sqrt((pad_pose.p[0] - mouse_pos.p[0])**2 +
                       (pad_pose.p[1] - mouse_pos.p[1])**2) < 0.1):
            pad_pose = rand_pose(xlim=pad_xlim, ylim=[0.05, 0.2], qpos=[1, 0, 0, 0],
                                 rotate_rand=False)

        colors = {"Red": (1, 0, 0), "Blue": (0, 0, 1), "Green": (0, 1, 0)}
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

        # --- Stage 2: Stapler (from press_stapler) on opposite side ---
        stapler_x_sign = -1 if mouse_pos.p[0] > 0 else 1
        stapler_xlim = [0.05, 0.2] if stapler_x_sign > 0 else [-0.2, -0.05]
        stapler_pos = rand_pose(
            xlim=stapler_xlim,
            ylim=[-0.1, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
        )
        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.stapler = create_actor(
            self,
            pose=stapler_pos,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=True,
        )
        self.add_prohibit_area(self.stapler, padding=0.05)

    def play_once(self):
        # === Stage 1: Place mouse on pad ===
        mouse_arm_tag = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.mouse, arm_tag=mouse_arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=mouse_arm_tag, z=0.1))
        self.move(
            self.place_actor(
                self.mouse,
                arm_tag=mouse_arm_tag,
                target_pose=self.pad_target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))
        self.move(self.back_to_origin(mouse_arm_tag))

        # === Stage 2: Press stapler with opposite arm ===
        stapler_arm_tag = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")
        self.move(
            self.grasp_actor(self.stapler, arm_tag=stapler_arm_tag, pre_grasp_dis=0.1,
                             grasp_dis=0.1, contact_point_id=2))
        self.move(self.close_gripper(arm_tag=stapler_arm_tag))
        self.move(
            self.grasp_actor(self.stapler, arm_tag=stapler_arm_tag, pre_grasp_dis=0.02,
                             grasp_dis=0.02, contact_point_id=2))

        self.info["info"] = {
            "{A}": f"047_mouse/base{self.mouse_id}",
            "{B}": f"{self.color_name}",
            "{C}": f"048_stapler/base{self.stapler_id}",
        }
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps = [0.03, 0.03]
        for position in positions:
            if (np.all(np.abs(position[:2] - stapler_pose[:2]) < eps)
                    and abs(position[2] - stapler_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False
