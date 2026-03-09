from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np
from copy import deepcopy
import math


class shake_then_place_bottle(Base_Task):
    """
    Stage 1: Shake one bottle horizontally.
    Stage 2: Put all bottles into the dustbin.
    Uses put_bottles_dustbin's table_xy_bias=[0.3, 0] and 114_bottle model.
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        pose_lst = []

        def create_bottle(model_id):
            bottle_pose = rand_pose(
                xlim=[-0.25, 0.3],
                ylim=[0.03, 0.23],
                rotate_rand=False,
                rotate_lim=[0, 1, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
            tag = True
            gen_lim = 100
            i = 1
            while tag and i < gen_lim:
                tag = False
                if np.abs(bottle_pose.p[0]) < 0.05:
                    tag = True
                for pose in pose_lst:
                    if np.sum(np.power(np.array(pose[:2]) - np.array(bottle_pose.p[:2]), 2)) < 0.0169:
                        tag = True
                        break
                if tag:
                    i += 1
                    bottle_pose = rand_pose(
                        xlim=[-0.25, 0.3],
                        ylim=[0.03, 0.23],
                        rotate_rand=False,
                        rotate_lim=[0, 1, 0],
                        qpos=[0.707, 0.707, 0, 0],
                    )
            pose_lst.append(bottle_pose.p[:2])
            bottle = create_actor(
                self,
                bottle_pose,
                modelname="114_bottle",
                convex=True,
                model_id=model_id,
            )
            return bottle

        self.bottles = []
        self.bottle_id = [1, 2, 3]
        self.bottle_num = 3
        for i in range(self.bottle_num):
            bottle = create_bottle(self.bottle_id[i])
            self.bottles.append(bottle)
            self.add_prohibit_area(bottle, padding=0.1)

        self.dustbin = create_actor(
            self.scene,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        self.delay(2)
        self.right_middle_pose = [0, 0.0, 0.88, 0, 1, 0, 0]

    def _shake_bottle(self, bottle):
        """Perform a horizontal shake motion on a bottle."""
        arm_tag = ArmTag("right" if bottle.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))

        target_quat = [0.707, 0, 0, 0.707]
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, quat=target_quat))

        # Two shake orientations (horizontal tilt)
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 8) * 7, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, deepcopy(target_quat))
        quat1 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        y_rotation = t3d.euler.euler2quat(0, -7 * (np.pi / 8), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, deepcopy(target_quat))
        quat2 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0., quat=quat1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.03, quat=quat2))
        for _ in range(2):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, quat=quat1))
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.05, quat=quat2))
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))

        # Return bottle to table
        self.move(
            self.place_actor(
                bottle,
                arm_tag=arm_tag,
                target_pose=bottle.get_pose().p.tolist() + [0.707, 0.707, 0, 0],
                pre_dis=0.0,
                dis=0.0,
                constrain="free",
            ))
        self.move(self.open_gripper(arm_tag))
        self.move(self.back_to_origin(arm_tag))

    def play_once(self):
        bottle_lst = sorted(self.bottles, key=lambda x: [x.get_pose().p[0] > 0, x.get_pose().p[1]])

        # === Stage 1: Shake the first bottle ===
        self._shake_bottle(bottle_lst[0])

        # === Stage 2: Put all bottles in dustbin ===
        delta_dis = 0.06
        left_end_action = Action("left", "move", [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65])

        for i in range(self.bottle_num):
            bottle = bottle_lst[i]
            arm_tag = ArmTag("left" if bottle.get_pose().p[0] < 0 else "right")

            if arm_tag == "left":
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
                self.move(self.move_by_displacement(arm_tag, z=0.1))
                self.move((ArmTag("left"), [left_end_action]))
            else:
                right_action = self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1)
                right_action[1][0].target_pose[2] += delta_dis
                right_action[1][1].target_pose[2] += delta_dis
                self.move(right_action, self.back_to_origin("left"))
                self.move(self.move_by_displacement(arm_tag, z=0.1))
                self.move(
                    self.place_actor(
                        bottle,
                        target_pose=self.right_middle_pose,
                        arm_tag=arm_tag,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                        constrain="align",
                    ))
                left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)
                left_action[1][0].target_pose[2] -= delta_dis
                left_action[1][1].target_pose[2] -= delta_dis
                self.move(left_action)
                self.move(self.open_gripper(ArmTag("right")))
                self.move((ArmTag("left"), [left_end_action]), self.back_to_origin("right"))
            self.move(self.open_gripper("left"))

        self.info["info"] = {
            "{A}": f"114_bottle/base{self.bottle_id[0]}",
            "{B}": f"114_bottle/base{self.bottle_id[1]}",
            "{C}": f"114_bottle/base{self.bottle_id[2]}",
        }
        return self.info

    def check_success(self):
        target_pose = [-0.45, 0]
        eps = np.array([0.221, 0.325])
        for bottle in self.bottles:
            bottle_pose = bottle.get_pose().p
            if (np.all(np.abs(bottle_pose[:2] - target_pose) < eps)
                    and bottle_pose[2] > 0.2 and bottle_pose[2] < 0.7):
                continue
            return False
        return True
