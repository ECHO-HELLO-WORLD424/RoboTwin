from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class handover_then_hang_mug(Base_Task):
    """
    Stage 1: Handover a microphone between arms (from handover_mic).
    Stage 2: Hang a mug on a rack (from hanging_mug).
    Mic in front area, mug+rack in back area, well separated.
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1: Microphone (from handover_mic) in front area ---
        # Place mic clearly on one side (|x| >= 0.15) by sampling from non-overlapping ranges
        mic_side = np.random.choice([-1, 1])
        rand_pos = rand_pose(
            xlim=[mic_side * 0.15, mic_side * 0.2],
            ylim=[-0.15, -0.05],
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
        self.grasp_arm_tag = ArmTag("right" if rand_pos.p[0] > 0 else "left")
        self.handover_arm_tag = self.grasp_arm_tag.opposite

        # --- Stage 2: Mug + rack (from hanging_mug) in back area ---
        # Place mug on the opposite side from the mic to minimise planner conflicts
        mug_xside = -1 if rand_pos.p[0] > 0 else 1
        self.mug_id = np.random.choice([i for i in range(10)])
        mug_xlim = sorted([mug_xside * 0.1, mug_xside * 0.25])
        mug_pose = rand_pose(
            xlim=mug_xlim,
            ylim=[0.05, 0.15],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
        )
        self.mug = create_actor(
            scene=self,
            pose=mug_pose,
            modelname="039_mug",
            convex=True,
            model_id=self.mug_id,
        )

        rack_xside = -mug_xside
        rack_xlim = sorted([rack_xside * 0.1, rack_xside * 0.3])
        rack_pose = rand_pose(
            xlim=rack_xlim,
            ylim=[0.1, 0.2],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            qpos=[-0.22, -0.22, 0.67, 0.67],
        )
        self.rack = create_actor(
            self, pose=rack_pose, modelname="040_rack", is_static=True, convex=True,
        )
        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)
        self.mug_middle_pos = [0.0, -0.15, 0.75 + self.table_z_bias, 1, 0, 0, 0]

    def play_once(self):
        # === Stage 1: Handover microphone ===
        grasp_arm_tag = ArmTag("right" if self.microphone.get_pose().p[0] > 0 else "left")
        handover_arm_tag = grasp_arm_tag.opposite

        self.move(
            self.grasp_actor(
                self.microphone, arm_tag=grasp_arm_tag,
                contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15],
                pre_grasp_dis=0.1,
            ))
        self.move(
            self.move_by_displacement(
                grasp_arm_tag, z=0.12,
                quat=(GRASP_DIRECTION_DIC["front_right"]
                      if grasp_arm_tag == "left" else GRASP_DIRECTION_DIC["front_left"]),
                move_axis="arm",
            ))
        self.move(
            self.place_actor(
                self.microphone, arm_tag=grasp_arm_tag,
                target_pose=self.handover_middle_pose,
                functional_point_id=0, pre_dis=0.0, dis=0.0,
                is_open=False, constrain="free",
            ))
        self.move(
            self.grasp_actor(
                self.microphone, arm_tag=handover_arm_tag,
                contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8],
                pre_grasp_dis=0.1,
            ))
        self.move(self.open_gripper(grasp_arm_tag))
        self.move(
            self.move_by_displacement(grasp_arm_tag, z=0.07, move_axis="arm"),
            self.move_by_displacement(
                handover_arm_tag,
                x=0.05 if handover_arm_tag == "right" else -0.05,
            ),
        )

        # Return both arms before mug task
        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        # === Stage 2: Hang mug on rack ===
        mug_grasp_arm = ArmTag("left" if self.mug.get_pose().p[0] < 0 else "right")
        mug_hang_arm = mug_grasp_arm.opposite

        self.move(self.grasp_actor(self.mug, arm_tag=mug_grasp_arm, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=mug_grasp_arm, z=0.08))
        self.move(
            self.place_actor(
                self.mug, arm_tag=mug_grasp_arm, target_pose=self.mug_middle_pos,
                pre_dis=0.05, dis=0.0, constrain="free",
            ))
        self.move(self.move_by_displacement(arm_tag=mug_grasp_arm, z=0.1))
        self.move(
            self.back_to_origin(mug_grasp_arm),
            self.grasp_actor(self.mug, arm_tag=mug_hang_arm, pre_grasp_dis=0.05),
        )
        self.move(
            self.move_by_displacement(
                arm_tag=mug_hang_arm, z=0.1, quat=GRASP_DIRECTION_DIC['front'],
            ))
        target_pose = self.rack.get_functional_point(0)
        self.move(
            self.place_actor(
                self.mug, arm_tag=mug_hang_arm, target_pose=target_pose,
                functional_point_id=0, constrain="align",
                pre_dis=0.05, dis=-0.05, pre_dis_axis='fp',
            ))
        self.move(self.move_by_displacement(arm_tag=mug_hang_arm, z=0.1, move_axis='arm'))

        self.info["info"] = {
            "{A}": f"018_microphone/base{self.microphone_id}",
            "{B}": f"039_mug/base{self.mug_id}",
            "{C}": "040_rack/base0",
        }
        return self.info

    def check_success(self):
        mug_fp = self.mug.get_functional_point(0)[:3]
        rack_pose = self.rack.get_pose().p
        rack_fp = self.rack.get_functional_point(0)[:3]
        rack_mid = (rack_pose + rack_fp) / 2
        return (np.all(abs((mug_fp - rack_mid)[:2]) < 0.02)
                and self.is_right_gripper_open()
                and mug_fp[2] > 0.86 + self.table_z_bias)
