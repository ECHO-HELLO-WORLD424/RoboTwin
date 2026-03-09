from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np
from copy import deepcopy


class dump_bin_then_sort_by_color(Base_Task):
    """
    Stage 1: Dump desk bin into large dustbin.
    Stage 2: Rank three colored blocks left-to-right (R < G < B).
    Uses table_xy_bias=[0.3, 0] from dump_bin_bigbin.
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        # --- Stage 1: Dustbin + Desk bin + garbage (from dump_bin_bigbin) ---
        self.dustbin = create_actor(
            self,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        deskbin_pose = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.2, -0.05],
            qpos=[0.651892, 0.651428, 0.274378, 0.274584],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 8.5, 0],
        )
        while abs(deskbin_pose.p[0]) < 0.05:
            deskbin_pose = rand_pose(
                xlim=[-0.2, 0.2],
                ylim=[-0.2, -0.05],
                qpos=[0.651892, 0.651428, 0.274378, 0.274584],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 8.5, 0],
            )
        self.deskbin_id = np.random.choice([0, 3, 7, 8, 9, 10], 1)[0]
        self.deskbin = create_actor(
            self,
            pose=deskbin_pose,
            modelname="063_tabletrashbin",
            model_id=self.deskbin_id,
            convex=True,
        )
        self.garbage_num = 5
        self.sphere_lst = []
        for i in range(self.garbage_num):
            sphere_pose = sapien.Pose(
                [
                    deskbin_pose.p[0] + np.random.rand() * 0.02 - 0.01,
                    deskbin_pose.p[1] + np.random.rand() * 0.02 - 0.01,
                    0.78 + i * 0.005,
                ],
                [1, 0, 0, 0],
            )
            sphere = create_sphere(
                self.scene,
                pose=sphere_pose,
                radius=0.008,
                color=[1, 0, 0],
                name="garbage",
            )
            self.sphere_lst.append(sphere)
            self.sphere_lst[-1].find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent).mass = 0.0001

        self.add_prohibit_area(self.deskbin, padding=0.04)
        self.prohibited_area.append([-0.2, -0.2, 0.2, 0.2])

        # Pour actions for dump stage
        action_lst = [
            Action(
                ArmTag('left'),
                "move",
                [-0.45, -0.05, 1.05, -0.694654, -0.178228, 0.165979, -0.676862],
            ),
            Action(
                ArmTag('left'),
                "move",
                [
                    -0.45,
                    -0.05 - np.random.rand() * 0.02,
                    1.05 - np.random.rand() * 0.02,
                    -0.694654,
                    -0.178228,
                    0.165979,
                    -0.676862,
                ],
            ),
        ]
        self.pour_actions = (ArmTag('left'), action_lst)
        self.middle_pose = [0, -0.1, 0.741 + self.table_z_bias, 1, 0, 0, 0]

        # --- Stage 2: Three colored blocks (from blocks_ranking_rgb) ---
        while True:
            block_pose_lst = []
            for i in range(3):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[0.08, 0.2],
                    zlim=[0.765],
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

                while (abs(block_pose.p[0]) < 0.05 or not check_block_pose(block_pose)):
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[0.08, 0.2],
                        zlim=[0.765],
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

        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)
        self.block1 = create_box(scene=self, pose=block_pose_lst[0], half_size=half_size,
                                 color=(1, 0, 0), name="box")
        self.block2 = create_box(scene=self, pose=block_pose_lst[1], half_size=half_size,
                                 color=(0, 1, 0), name="box")
        self.block3 = create_box(scene=self, pose=block_pose_lst[2], half_size=half_size,
                                 color=(0, 0, 1), name="box")

        y_pose = np.random.uniform(0.25, 0.3)
        self.block1_target_pose = [np.random.uniform(-0.09, -0.08), y_pose,
                                   0.74 + self.table_z_bias] + [0, 1, 0, 0]
        self.block2_target_pose = [np.random.uniform(-0.01, 0.01), y_pose,
                                   0.74 + self.table_z_bias] + [0, 1, 0, 0]
        self.block3_target_pose = [np.random.uniform(0.08, 0.09), y_pose,
                                   0.74 + self.table_z_bias] + [0, 1, 0, 0]
        self.last_gripper = None

    def _pick_and_place_block(self, block, target_pose):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01),
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
        # === Stage 1: Dump desk bin into large dustbin ===
        deskbin_pose = self.deskbin.get_pose().p
        grasp_arm = ArmTag("left" if deskbin_pose[0] < 0 else "right")
        place_arm = ArmTag("left")

        if grasp_arm == "right":
            self.move(
                self.grasp_actor(self.deskbin, arm_tag=grasp_arm, pre_grasp_dis=0.08,
                                 contact_point_id=3))
            self.move(self.move_by_displacement(grasp_arm, z=0.08, move_axis="arm"))
            self.move(
                self.place_actor(self.deskbin, target_pose=self.middle_pose, arm_tag=grasp_arm,
                                 pre_dis=0.08, dis=0.01))
            self.move(self.move_by_displacement(grasp_arm, z=0.1, move_axis="arm"))
            self.move(
                self.back_to_origin(grasp_arm),
                self.grasp_actor(self.deskbin, arm_tag=place_arm, pre_grasp_dis=0.08,
                                 contact_point_id=1),
            )
        else:
            self.move(
                self.grasp_actor(self.deskbin, arm_tag=place_arm, pre_grasp_dis=0.08,
                                 contact_point_id=1))

        self.move(self.move_by_displacement(arm_tag=place_arm, z=0.08, move_axis="arm"))
        for _ in range(3):
            self.move(self.pour_actions)
        self.delay(6)

        # Return arms before sorting
        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        # === Stage 2: Sort blocks by color (R→G→B, left to right) ===
        self.last_gripper = None
        self._pick_and_place_block(self.block1, self.block1_target_pose)
        self._pick_and_place_block(self.block2, self.block2_target_pose)
        self._pick_and_place_block(self.block3, self.block3_target_pose)

        self.info["info"] = {
            "{A}": f"063_tabletrashbin/base{self.deskbin_id}",
            "{B}": "red block",
            "{C}": "green block",
            "{D}": "blue block",
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
