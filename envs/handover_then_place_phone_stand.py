from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class handover_then_place_phone_stand(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1 objects: Microphone (from handover_mic) ---
        rand_pos = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.05, 0.0],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
        )
        while abs(rand_pos.p[0]) < 0.15:
            rand_pos = rand_pose(
                xlim=[-0.2, 0.2],
                ylim=[-0.05, 0.0],
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )
        self.microphone_id = np.random.choice([0, 4, 5], 1)[0]
        self.microphone = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="018_microphone",
            convex=True,
            model_id=self.microphone_id,
        )
        self.add_prohibit_area(self.microphone, padding=0.07)
        self.handover_middle_pose = [0, -0.05, 0.98, 0, 1, 0, 0]
        # Determine arm assignment based on mic position
        self.grasp_arm_tag = ArmTag("right" if self.microphone.get_pose().p[0] > 0 else "left")
        self.handover_arm_tag = self.grasp_arm_tag.opposite

        # --- Stage 2 objects: Phone + Stand (from place_phone_stand) ---
        # Phone goes on the same side as the grasp arm (that arm will be free after handover)
        grasp_x_sign = 1 if self.grasp_arm_tag == "right" else -1
        phone_x_lim = [0.05, 0.25] if grasp_x_sign > 0 else [-0.25, -0.05]
        stand_x_lim = [0.0, 0.15] if grasp_x_sign > 0 else [-0.15, 0.0]

        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]
        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        phone_pose = rand_pose(
            xlim=phone_x_lim,
            ylim=[-0.2, 0.0],
            qpos=ori_quat[self.phone_id],
            rotate_rand=True,
            rotate_lim=[0, 0.7, 0],
        )
        self.phone = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
        )
        self.phone.set_mass(0.01)

        stand_pose = rand_pose(
            xlim=stand_x_lim,
            ylim=[0.05, 0.2],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
        )
        while np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2])**2)) < 0.15:
            stand_pose = rand_pose(
                xlim=stand_x_lim,
                ylim=[0.05, 0.2],
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )
        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.stand = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=True,
        )
        self.add_prohibit_area(self.phone, padding=0.15)
        self.add_prohibit_area(self.stand, padding=0.15)

    def play_once(self):
        grasp_arm_tag = ArmTag("right" if self.microphone.get_pose().p[0] > 0 else "left")
        handover_arm_tag = grasp_arm_tag.opposite

        # === Stage 1: Handover microphone ===
        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=grasp_arm_tag,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                pre_grasp_dis=0.1,
            ))
        self.move(
            self.move_by_displacement(
                grasp_arm_tag,
                z=0.12,
                quat=(GRASP_DIRECTION_DIC["front_right"]
                      if grasp_arm_tag == "left" else GRASP_DIRECTION_DIC["front_left"]),
                move_axis="arm",
            ))
        self.move(
            self.place_actor(
                self.microphone,
                arm_tag=grasp_arm_tag,
                target_pose=self.handover_middle_pose,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            ))
        self.move(
            self.grasp_actor(
                self.microphone,
                arm_tag=handover_arm_tag,
                contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8],
                pre_grasp_dis=0.1,
            ))
        self.move(self.open_gripper(grasp_arm_tag))
        self.move(
            self.move_by_displacement(grasp_arm_tag, z=0.07, move_axis="arm"),
            self.move_by_displacement(handover_arm_tag,
                                      x=0.05 if handover_arm_tag == "right" else -0.05),
        )

        # === Stage 2: Place phone on stand ===
        # grasp_arm_tag is now free; phone is on the same side
        phone_arm_tag = ArmTag("left" if self.phone.get_pose().p[0] < 0 else "right")
        self.move(self.back_to_origin(phone_arm_tag))
        self.move(self.grasp_actor(self.phone, arm_tag=phone_arm_tag, pre_grasp_dis=0.08))
        stand_func_pose = self.stand.get_functional_point(0)
        self.move(
            self.place_actor(
                self.phone,
                arm_tag=phone_arm_tag,
                target_pose=stand_func_pose,
                functional_point_id=0,
                dis=0,
                constrain="align",
            ))

        self.info["info"] = {
            "{A}": f"018_microphone/base{self.microphone_id}",
            "{B}": f"077_phone/base{self.phone_id}",
            "{C}": f"078_phonestand/base{self.stand_id}",
        }
        return self.info

    def check_success(self):
        phone_func_pose = np.array(self.phone.get_functional_point(0))
        stand_func_pose = np.array(self.stand.get_functional_point(0))
        eps = np.array([0.045, 0.04, 0.04])
        return np.all(np.abs(phone_func_pose - stand_func_pose)[:3] < eps)
