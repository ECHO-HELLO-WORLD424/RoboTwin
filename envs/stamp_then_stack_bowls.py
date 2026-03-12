from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np
from copy import deepcopy


class stamp_then_stack_bowls(Base_Task):
    """
    Stage 1: Stamp a seal on a colored target (from stamp_seal).
    Stage 2: Stack two bowls (from stack_bowls_two).
    Seal+pad in front area (y < 0), bowls in back area (y > 0.05).
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1: Seal + target pad (from stamp_seal), in front area ---
        seal_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, -0.08],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        while abs(seal_pose.p[0]) < 0.05:
            seal_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, -0.08],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
        self.seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]
        self.seal = create_actor(
            scene=self,
            pose=seal_pose,
            modelname="100_seal",
            convex=True,
            model_id=self.seal_id,
        )
        self.seal.set_mass(0.05)

        # Target pad on the same side as the seal, also in front area
        pad_xlim = [0.05, 0.25] if seal_pose.p[0] > 0 else [-0.25, -0.05]
        target_pose = rand_pose(
            xlim=pad_xlim,
            ylim=[-0.2, -0.08],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        while np.sqrt((target_pose.p[0] - seal_pose.p[0])**2
                       + (target_pose.p[1] - seal_pose.p[1])**2) < 0.1:
            target_pose = rand_pose(
                xlim=pad_xlim,
                ylim=[-0.2, -0.08],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )

        colors = {"Red": (1, 0, 0), "Green": (0, 1, 0), "Blue": (0, 0, 1)}
        color_items = list(colors.items())
        self.color_name, self.color_value = color_items[np.random.choice(len(color_items))]
        self.target = create_visual_box(
            scene=self,
            pose=target_pose,
            half_size=[0.035, 0.035, 0.0005],
            color=self.color_value,
            name="box",
        )
        # Only protect the front area — bowls go in the back
        self.prohibited_area.append([-0.3, -0.25, 0.3, 0.0])

        # --- Stage 2: Two bowls (from stack_bowls_two), in back area only ---
        bowl_pose_lst = []
        for i in range(2):
            bowl_pose = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[0.05, 0.2],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )

            def check_bowl_pose(bp):
                for j in range(len(bowl_pose_lst)):
                    if np.sum(pow(bp.p[:2] - bowl_pose_lst[j].p[:2], 2)) < 0.0169:
                        return False
                return True

            while abs(bowl_pose.p[0]) < 0.09 or not check_bowl_pose(bowl_pose):
                bowl_pose = rand_pose(
                    xlim=[-0.3, 0.3],
                    ylim=[0.05, 0.2],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=False,
                )
            bowl_pose_lst.append(deepcopy(bowl_pose))

        bowl_pose_lst = sorted(bowl_pose_lst, key=lambda x: x.p[1])
        self.bowl1 = create_actor(
            self, pose=bowl_pose_lst[0], modelname="002_bowl", model_id=3, convex=True,
        )
        self.bowl2 = create_actor(
            self, pose=bowl_pose_lst[1], modelname="002_bowl", model_id=3, convex=True,
        )
        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        # Stack target in back area, away from seal/pad
        self.bowl1_target_pose = np.array([0, 0.1, 0.76])
        self.quat_of_target_pose = [0, 0.707, 0.707, 0]
        self.las_arm = None

    def _move_bowl(self, actor, target_pose):
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")
        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(
                self.grasp_actor(
                    actor, arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                ))
        else:
            self.move(
                self.grasp_actor(
                    actor, arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        self.move(self.move_by_displacement(arm_tag, z=0.1))
        self.move(
            self.place_actor(
                actor,
                target_pose=target_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.09))
        self.las_arm = arm_tag

    def play_once(self):
        # === Stage 1: Stamp the seal ===
        seal_arm = ArmTag("right" if self.seal.get_pose().p[0] > 0 else "left")
        self.move(
            self.grasp_actor(
                self.seal, arm_tag=seal_arm, pre_grasp_dis=0.1,
                contact_point_id=[4, 5, 6, 7],
            ))
        self.move(self.move_by_displacement(arm_tag=seal_arm, z=0.05))
        self.move(
            self.place_actor(
                self.seal, arm_tag=seal_arm,
                target_pose=self.target.get_pose(),
                pre_dis=0.1, constrain="auto",
            ))

        # Return arm before stacking
        self.move(self.back_to_origin(seal_arm))

        # === Stage 2: Stack two bowls ===
        self.las_arm = None
        self._move_bowl(self.bowl1, self.bowl1_target_pose)
        self._move_bowl(self.bowl2, self.bowl1.get_pose().p + [0, 0, 0.05])

        self.info["info"] = {
            "{A}": f"100_seal/base{self.seal_id}",
            "{B}": f"{self.color_name}",
        }
        return self.info

    def check_success(self):
        seal_pose = self.seal.get_pose().p
        target_pos = self.target.get_pose().p
        seal_ok = np.all(abs(seal_pose[:2] - target_pos[:2]) < 0.01)

        bowl1_pose = self.bowl1.get_pose().p
        bowl2_pose = self.bowl2.get_pose().p
        b_low, b_high = sorted([bowl1_pose, bowl2_pose], key=lambda x: x[2])
        target_height = [0.74 + self.table_z_bias, 0.77 + self.table_z_bias]
        bowl_ok = (np.all(abs(b_low[:2] - b_high[:2]) < 0.04)
                   and np.all(np.array([b_low[2], b_high[2]]) - target_height < 0.02)
                   and self.is_left_gripper_open()
                   and self.is_right_gripper_open())

        return seal_ok and bowl_ok
