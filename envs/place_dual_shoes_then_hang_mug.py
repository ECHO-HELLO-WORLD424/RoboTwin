from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class place_dual_shoes_then_hang_mug(Base_Task):
    """
    Stage 1: Place both shoes into the shoe box.
    Stage 2: Hang the mug on the rack.
    Uses table_height_bias=-0.1 from place_dual_shoes.
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_height_bias=-0.1, **kwags)

    def load_actors(self):
        # --- Stage 1: Shoe box + two shoes (from place_dual_shoes) ---
        self.shoe_box = create_actor(
            self,
            pose=sapien.Pose([0, -0.13, 0.74], [0.5, 0.5, -0.5, -0.5]),
            modelname="007_shoe-box",
            convex=True,
            is_static=True,
        )

        shoe_id = np.random.choice([i for i in range(10)])
        self.shoe_id = shoe_id

        # Left shoe
        shoes_pose = rand_pose(
            xlim=[-0.3, -0.2],
            ylim=[-0.1, 0.05],
            zlim=[0.741],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )
        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[-0.3, -0.2],
                ylim=[-0.1, 0.05],
                zlim=[0.741],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
        self.left_shoe = create_actor(
            self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=shoe_id,
        )

        # Right shoe
        shoes_pose = rand_pose(
            xlim=[0.2, 0.3],
            ylim=[-0.1, 0.05],
            zlim=[0.741],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )
        while np.sum(pow(shoes_pose.get_p()[:2] - np.zeros(2), 2)) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[0.2, 0.3],
                ylim=[-0.1, 0.05],
                zlim=[0.741],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
        self.right_shoe = create_actor(
            self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=shoe_id,
        )

        self.add_prohibit_area(self.left_shoe, padding=0.02)
        self.add_prohibit_area(self.right_shoe, padding=0.02)
        self.prohibited_area.append([-0.15, -0.25, 0.15, 0.01])  # protect shoe box area

        # --- Stage 2: Mug + rack (from hanging_mug), placed in back area ---
        # Use back area (ylim > 0.08) to avoid shoe box at y=-0.13
        self.mug_id = np.random.choice([i for i in range(10)])
        self.mug = rand_create_actor(
            self,
            xlim=[-0.28, -0.15],
            ylim=[0.08, 0.2],
            ylim_prop=True,
            modelname="039_mug",
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
            qpos=[0.707, 0.707, 0, 0],
            convex=True,
            model_id=self.mug_id,
        )

        rack_pose = rand_pose(
            xlim=[0.1, 0.3],
            ylim=[0.1, 0.2],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            qpos=[-0.22, -0.22, 0.67, 0.67],
        )
        self.rack = create_actor(self, pose=rack_pose, modelname="040_rack", is_static=True, convex=True)

        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)
        # Intermediate pose for mug transfer (adapt z for table height bias)
        self.mug_middle_pos = [0.0, -0.15, 0.75 + self.table_z_bias, 1, 0, 0, 0]

    def play_once(self):
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        # === Stage 1: Place both shoes in the shoe box ===
        self.move(
            self.grasp_actor(self.left_shoe, arm_tag=left_arm_tag, pre_grasp_dis=0.1),
            self.grasp_actor(self.right_shoe, arm_tag=right_arm_tag, pre_grasp_dis=0.1),
        )
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.15),
            self.move_by_displacement(right_arm_tag, z=0.15),
        )
        left_target = self.shoe_box.get_functional_point(0)
        right_target = self.shoe_box.get_functional_point(1)
        left_place_pose = self.place_actor(
            self.left_shoe,
            target_pose=left_target,
            arm_tag=left_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        right_place_pose = self.place_actor(
            self.right_shoe,
            target_pose=right_target,
            arm_tag=right_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        self.move(
            left_place_pose,
            self.move_by_displacement(right_arm_tag, x=0.1, y=-0.05, quat=GRASP_DIRECTION_DIC["top_down"]),
        )
        self.move(self.back_to_origin(left_arm_tag), right_place_pose)
        self.delay(3)

        # Return both arms before mug task
        self.move(self.back_to_origin(left_arm_tag))
        self.move(self.back_to_origin(right_arm_tag))

        # === Stage 2: Hang mug on rack ===
        grasp_arm_tag = ArmTag("left")
        hang_arm_tag = ArmTag("right")

        self.move(self.grasp_actor(self.mug, arm_tag=grasp_arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.08))
        self.move(
            self.place_actor(self.mug, arm_tag=grasp_arm_tag, target_pose=self.mug_middle_pos,
                             pre_dis=0.05, dis=0.0, constrain="free"))
        self.move(self.move_by_displacement(arm_tag=grasp_arm_tag, z=0.1))
        self.move(
            self.back_to_origin(grasp_arm_tag),
            self.grasp_actor(self.mug, arm_tag=hang_arm_tag, pre_grasp_dis=0.05),
        )
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, quat=GRASP_DIRECTION_DIC['front']))

        target_pose = self.rack.get_functional_point(0)
        self.move(
            self.place_actor(self.mug, arm_tag=hang_arm_tag, target_pose=target_pose,
                             functional_point_id=0, constrain="align",
                             pre_dis=0.05, dis=-0.05, pre_dis_axis='fp'))
        self.move(self.move_by_displacement(arm_tag=hang_arm_tag, z=0.1, move_axis='arm'))

        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "007_shoe-box/base0",
            "{C}": f"039_mug/base{self.mug_id}",
            "{D}": "040_rack/base0",
        }
        return self.info

    def check_success(self):
        mug_function_pose = self.mug.get_functional_point(0)[:3]
        rack_pose = self.rack.get_pose().p
        rack_function_pose = self.rack.get_functional_point(0)[:3]
        rack_middle_pose = (rack_pose + rack_function_pose) / 2
        eps = 0.02
        return (np.all(abs((mug_function_pose - rack_middle_pose)[:2]) < eps)
                and self.is_right_gripper_open()
                and mug_function_pose[2] > 0.86 + self.table_z_bias)
