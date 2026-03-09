from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class open_laptop_then_place_object_inside(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1: Laptop (from open_laptop) ---
        self.laptop_model_name = "015_laptop"
        self.laptop_model_id = np.random.randint(0, 11)
        self.laptop = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.laptop_model_name,
            modelid=self.laptop_model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.1, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 3],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.2])
        self.laptop.set_mass(0.01)
        self.laptop.set_properties(1, 0)
        self.add_prohibit_area(self.laptop, padding=0.1)
        self.laptop_arm_tag = None  # set during play_once

        # --- Stage 2: Container + Plate (from place_container_plate) ---
        # Container on one side (away from laptop center)
        container_x_side = np.random.choice([-1, 1])
        container_xlim = [0.15, 0.28] if container_x_side > 0 else [-0.28, -0.15]

        container_pose = rand_pose(
            xlim=container_xlim,
            ylim=[-0.1, 0.05],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        id_list = {"002_bowl": [1, 2, 3, 5], "021_cup": [1, 2, 3, 4, 5, 6, 7]}
        self.actor_name = np.random.choice(["002_bowl", "021_cup"])
        self.container_id = np.random.choice(id_list[self.actor_name])
        self.container = create_actor(
            self,
            pose=container_pose,
            modelname=self.actor_name,
            model_id=self.container_id,
            convex=True,
        )

        x = 0.05 if self.container.get_pose().p[0] > 0 else -0.05
        self.plate_id = 0
        plate_pose = rand_pose(
            xlim=[x - 0.03, x + 0.03],
            ylim=[-0.2, -0.12],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        self.plate = create_actor(
            self,
            pose=plate_pose,
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True,
        )
        self.add_prohibit_area(self.container, padding=0.1)
        self.add_prohibit_area(self.plate, padding=0.1)

    def play_once(self):
        # === Stage 1: Open laptop ===
        face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        self.laptop_arm_tag = ArmTag("left" if face_prod > 0 else "right")

        self.move(
            self.grasp_actor(self.laptop, arm_tag=self.laptop_arm_tag, pre_grasp_dis=0.08,
                             contact_point_id=0))
        for _ in range(15):
            self.move(
                self.grasp_actor(
                    self.laptop,
                    arm_tag=self.laptop_arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=1,
                ))
            if not self.plan_success:
                break
            if self._laptop_open_enough():
                break

        self.move(self.back_to_origin(self.laptop_arm_tag))

        # === Stage 2: Place container on plate ===
        container_pose = self.container.get_pose().p
        arm_tag = ArmTag("right" if container_pose[0] > 0 else "left")

        self.move(
            self.grasp_actor(
                self.container,
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                pre_grasp_dis=0.1,
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
        self.move(
            self.place_actor(
                self.container,
                target_pose=self.plate.get_functional_point(0),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.12,
                dis=0.03,
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))

        self.info["info"] = {
            "{A}": f"{self.laptop_model_name}/base{self.laptop_model_id}",
            "{B}": f"{self.actor_name}/base{self.container_id}",
            "{C}": f"003_plate/base{self.plate_id}",
        }
        return self.info

    def _laptop_open_enough(self, target=0.4):
        limit = self.laptop.get_qlimits()[0]
        qpos = self.laptop.get_qpos()
        return qpos[0] >= limit[0] + (limit[1] - limit[0]) * target

    def check_success(self):
        container_pose = self.container.get_pose().p
        target_pose = self.plate.get_pose().p
        eps = np.array([0.05, 0.05, 0.03])
        return (np.all(abs(container_pose[:3] - target_pose) < eps) and self.is_left_gripper_open()
                and self.is_right_gripper_open())
