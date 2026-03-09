from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np
from copy import deepcopy


class stack_then_scan(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 2 objects first: Scanner + Tea-box (from scan_object) ---
        # Load these first so prohibit areas are set before placing blocks
        tag = np.random.randint(2)
        if tag == 0:
            scanner_x_lim = [-0.25, -0.05]
            object_x_lim = [0.05, 0.25]
        else:
            scanner_x_lim = [0.05, 0.25]
            object_x_lim = [-0.25, -0.05]

        scanner_pose = rand_pose(
            xlim=scanner_x_lim,
            ylim=[0.05, 0.15],
            qpos=[0, 0, 0.707, 0.707],
            rotate_rand=True,
            rotate_lim=[0, 1.2, 0],
        )
        self.scanner_id = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        self.scanner = create_actor(
            scene=self.scene,
            pose=scanner_pose,
            modelname="024_scanner",
            convex=True,
            model_id=self.scanner_id,
        )

        object_pose = rand_pose(
            xlim=object_x_lim,
            ylim=[0.05, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 1.2, 0],
        )
        self.object_id = np.random.choice([0, 1, 2, 3, 4, 5], 1)[0]
        self.object = create_actor(
            scene=self.scene,
            pose=object_pose,
            modelname="112_tea-box",
            convex=True,
            model_id=self.object_id,
        )
        self.add_prohibit_area(self.scanner, padding=0.12)
        self.add_prohibit_area(self.object, padding=0.12)
        self.left_object_target_pose = [-0.03, -0.02, 0.95, 0.707, 0, -0.707, 0]
        self.right_object_target_pose = [0.03, -0.02, 0.95, 0.707, 0, 0.707, 0]

        # --- Stage 1 objects: Two colored blocks (from stack_blocks_two) ---
        block_half_size = 0.025
        block_pose_lst = []
        for i in range(2):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.1, 0.0],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(bp):
                for j in range(len(block_pose_lst)):
                    if np.sum(pow(bp.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01:
                        return False
                return True

            while (abs(block_pose.p[0]) < 0.05
                   or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
                   or not check_block_pose(block_pose)):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.1, 0.0],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            block_pose_lst.append(deepcopy(block_pose))

        self.block1 = create_box(scene=self, pose=block_pose_lst[0],
                                 half_size=(block_half_size, block_half_size, block_half_size),
                                 color=(1, 0, 0), name="box")
        self.block2 = create_box(scene=self, pose=block_pose_lst[1],
                                 half_size=(block_half_size, block_half_size, block_half_size),
                                 color=(0, 1, 0), name="box")
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        # Target for stacking
        self.block1_target_pose = [0, -0.1, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        self.last_gripper = None
        self.last_actor = None

    def _pick_and_place_block(self, block):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        if self.last_actor is None:
            target_pose = [0, -0.1, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        else:
            target_pose = self.last_actor.get_functional_point(1)

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        # === Stage 1: Stack two blocks ===
        self._pick_and_place_block(self.block1)
        self._pick_and_place_block(self.block2)

        # Return arms to origin before scan
        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        # === Stage 2: Scan the object ===
        scanner_arm_tag = ArmTag("left" if self.scanner.get_pose().p[0] < 0 else "right")
        object_arm_tag = scanner_arm_tag.opposite

        self.move(
            self.grasp_actor(self.scanner, arm_tag=scanner_arm_tag, pre_grasp_dis=0.08),
            self.grasp_actor(self.object, arm_tag=object_arm_tag, pre_grasp_dis=0.08),
        )
        self.move(
            self.move_by_displacement(arm_tag=scanner_arm_tag,
                                      x=0.05 if scanner_arm_tag == "right" else -0.05, z=0.13),
            self.move_by_displacement(arm_tag=object_arm_tag,
                                      x=0.05 if object_arm_tag == "right" else -0.05, z=0.13),
        )
        object_target_pose = (self.right_object_target_pose
                              if object_arm_tag == "right" else self.left_object_target_pose)
        self.move(
            self.place_actor(
                self.object,
                arm_tag=object_arm_tag,
                target_pose=object_target_pose,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ))
        self.move(
            self.place_actor(
                self.scanner,
                arm_tag=scanner_arm_tag,
                target_pose=self.object.get_functional_point(1),
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.05,
                is_open=False,
            ))

        self.info["info"] = {
            "{A}": f"112_tea-box/base{self.object_id}",
            "{B}": f"024_scanner/base{self.scanner_id}",
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        scanner_func_pose = self.scanner.get_functional_point(0)
        target_vec = t3d.quaternions.quat2mat(scanner_func_pose[-4:]) @ np.array([0, 0, -1])
        obj2scanner_vec = scanner_func_pose[:3] - object_pose
        dis = np.sum(target_vec * obj2scanner_vec)
        object_pose1 = object_pose + dis * target_vec
        eps = 0.025
        return (np.all(np.abs(object_pose1 - scanner_func_pose[:3]) < eps) and dis > 0 and dis < 0.07
                and self.is_left_gripper_close() and self.is_right_gripper_close())
