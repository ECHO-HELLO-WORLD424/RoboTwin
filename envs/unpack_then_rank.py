from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np
from copy import deepcopy


class unpack_then_rank(Base_Task):
    """
    Stage 1: Scan an object ("unpack" / barcode scan).
    Stage 2: Rank three blocks by size (large→medium→small, left to right).
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 2 objects first: Blocks (from blocks_ranking_size) ---
        # Use back area so they don't conflict with scanner/tea-box
        halfsize_lst = [
            np.random.uniform(0.03, 0.033),
            np.random.uniform(0.024, 0.027),
            np.random.uniform(0.018, 0.021),
        ]
        color_lst = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(3)]
        while True:
            block_pose_lst = []
            for i in range(3):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[0.08, 0.2],
                    zlim=[0.741 + halfsize_lst[i]],
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

                while abs(block_pose.p[0]) < 0.05 or not check_block_pose(block_pose):
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[0.08, 0.2],
                        zlim=[0.741 + halfsize_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                block_pose_lst.append(deepcopy(block_pose))

            b1, b2, b3 = block_pose_lst[0].p, block_pose_lst[1].p, block_pose_lst[2].p
            eps = [0.12, 0.03]
            if (np.all(abs(b1[:2] - b2[:2]) < eps) and np.all(abs(b2[:2] - b3[:2]) < eps)
                    and b1[0] < b2[0] and b2[0] < b3[0]):
                continue
            break

        self.block1 = create_box(scene=self, pose=block_pose_lst[0],
                                 half_size=(halfsize_lst[0],) * 3, color=color_lst[0], name="box")
        self.block2 = create_box(scene=self, pose=block_pose_lst[1],
                                 half_size=(halfsize_lst[1],) * 3, color=color_lst[1], name="box")
        self.block3 = create_box(scene=self, pose=block_pose_lst[2],
                                 half_size=(halfsize_lst[2],) * 3, color=color_lst[2], name="box")

        self.add_prohibit_area(self.block1, padding=0.1)
        self.add_prohibit_area(self.block2, padding=0.1)
        self.add_prohibit_area(self.block3, padding=0.1)
        self.prohibited_area.append([-0.27, 0.06, 0.27, 0.22])  # protect block area for scan stage

        y_pose = np.random.uniform(0.25, 0.3)
        self.block1_target_pose = [np.random.uniform(-0.1, -0.09), y_pose,
                                   0.74 + self.table_z_bias] + [0, 1, 0, 0]
        self.block2_target_pose = [np.random.uniform(0.01, 0.02), y_pose,
                                   0.74 + self.table_z_bias] + [0, 1, 0, 0]
        self.block3_target_pose = [np.random.uniform(0.08, 0.09), y_pose,
                                   0.74 + self.table_z_bias] + [0, 1, 0, 0]

        # --- Stage 1 objects: Scanner + Tea-box (from scan_object) ---
        # Place in the front area
        tag = np.random.randint(2)
        scanner_x_lim = [-0.25, -0.05] if tag == 0 else [0.05, 0.25]
        object_x_lim = [0.05, 0.25] if tag == 0 else [-0.25, -0.05]

        scanner_pose = rand_pose(
            xlim=scanner_x_lim,
            ylim=[-0.15, -0.05],
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
            ylim=[-0.2, -0.05],
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
        self.add_prohibit_area(self.scanner, padding=0.1)
        self.add_prohibit_area(self.object, padding=0.1)
        self.left_object_target_pose = [-0.03, -0.02, 0.95, 0.707, 0, -0.707, 0]
        self.right_object_target_pose = [0.03, -0.02, 0.95, 0.707, 0, 0.707, 0]
        self.last_gripper = None

    def _pick_and_place_block(self, block, target_pose):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self.move(
            self.place_actor(block, target_pose=target_pose, arm_tag=arm_tag,
                             functional_point_id=0, pre_dis=0.09, dis=0.02, constrain="align"))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        self.last_gripper = arm_tag

    def play_once(self):
        # === Stage 1: Scan the object ===
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
            self.place_actor(self.object, arm_tag=object_arm_tag, target_pose=object_target_pose,
                             pre_dis=0.0, dis=0.0, is_open=False))
        self.move(
            self.place_actor(self.scanner, arm_tag=scanner_arm_tag,
                             target_pose=self.object.get_functional_point(1),
                             functional_point_id=0, pre_dis=0.05, dis=0.05, is_open=False))

        # Return arms and set objects down before ranking
        self.move(self.open_gripper(scanner_arm_tag))
        self.move(self.open_gripper(object_arm_tag))
        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        # === Stage 2: Rank blocks by size (large→medium→small, left→right) ===
        self.last_gripper = None
        self._pick_and_place_block(self.block3, self.block3_target_pose)
        self._pick_and_place_block(self.block2, self.block2_target_pose)
        self._pick_and_place_block(self.block1, self.block1_target_pose)

        self.info["info"] = {
            "{A}": f"112_tea-box/base{self.object_id}",
            "{B}": f"024_scanner/base{self.scanner_id}",
        }
        return self.info

    def check_success(self):
        b1 = self.block1.get_pose().p
        b2 = self.block2.get_pose().p
        b3 = self.block3.get_pose().p
        eps = [0.13, 0.03]
        return (np.all(abs(b1[:2] - b2[:2]) < eps) and np.all(abs(b2[:2] - b3[:2]) < eps)
                and b1[0] < b2[0] and b2[0] < b3[0] and self.is_left_gripper_open()
                and self.is_right_gripper_open())
