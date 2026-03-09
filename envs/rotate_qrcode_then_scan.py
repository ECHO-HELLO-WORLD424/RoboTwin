from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np


class rotate_qrcode_then_scan(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Stage 1: QR code (from rotate_qrcode) ---
        # Place QR code on one side, front area
        qrcode_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, -0.05],
            qpos=[0, 0, 0.707, 0.707],
            rotate_rand=True,
            rotate_lim=[0, 0.7, 0],
        )
        while abs(qrcode_pose.p[0]) < 0.12:
            qrcode_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, -0.05],
                qpos=[0, 0, 0.707, 0.707],
                rotate_rand=True,
                rotate_lim=[0, 0.7, 0],
            )
        self.qrcode_model_id = np.random.choice([0, 1, 2, 3], 1)[0]
        self.qrcode = create_actor(
            self,
            pose=qrcode_pose,
            modelname="070_paymentsign",
            convex=True,
            model_id=self.qrcode_model_id,
        )
        self.add_prohibit_area(self.qrcode, padding=0.12)
        qrcode_x = self.qrcode.get_pose().p[0]
        target_x = -0.2 if qrcode_x < 0 else 0.2
        self.qrcode_target_pose = [target_x, -0.15, 0.74 + self.table_z_bias, 1, 0, 0, 0]

        # --- Stage 2: Scanner + Tea-box (from scan_object) ---
        # Place on the back half of the table to avoid QR code area
        scanner_x_lim = [-0.25, -0.05] if qrcode_x > 0 else [0.05, 0.25]
        object_x_lim = [0.05, 0.25] if qrcode_x > 0 else [-0.25, -0.05]

        scanner_pose = rand_pose(
            xlim=scanner_x_lim,
            ylim=[0.03, 0.15],
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
            ylim=[0.03, 0.15],
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

    def play_once(self):
        # === Stage 1: Rotate QR code to face-up orientation ===
        qrcode_arm_tag = ArmTag("left" if self.qrcode.get_pose().p[0] < 0 else "right")
        self.move(self.grasp_actor(self.qrcode, arm_tag=qrcode_arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=qrcode_arm_tag, z=0.07))
        self.move(
            self.place_actor(
                self.qrcode,
                arm_tag=qrcode_arm_tag,
                target_pose=self.qrcode_target_pose,
                pre_dis=0.07,
                dis=0.01,
                constrain="align",
            ))
        self.move(self.back_to_origin(qrcode_arm_tag))

        # === Stage 2: Scan the tea-box with the scanner ===
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
            "{A}": f"070_paymentsign/base{self.qrcode_model_id}",
            "{B}": f"112_tea-box/base{self.object_id}",
            "{C}": f"024_scanner/base{self.scanner_id}",
        }
        return self.info

    def check_success(self):
        # Check QR code is face-up
        qrcode_quat = self.qrcode.get_pose().q
        if qrcode_quat[0] < 0:
            qrcode_quat = qrcode_quat * -1
        target_quat = [0.707, 0.707, 0, 0]
        qrcode_ok = (np.all(np.abs(qrcode_quat - target_quat) < 0.05)
                     and self.qrcode.get_pose().p[2] < 0.75 + self.table_z_bias)

        # Check scanning is active
        object_pose = self.object.get_pose().p
        scanner_func_pose = self.scanner.get_functional_point(0)
        target_vec = t3d.quaternions.quat2mat(scanner_func_pose[-4:]) @ np.array([0, 0, -1])
        obj2scanner_vec = scanner_func_pose[:3] - object_pose
        dis = np.sum(target_vec * obj2scanner_vec)
        object_pose1 = object_pose + dis * target_vec
        eps = 0.025
        scan_ok = (np.all(np.abs(object_pose1 - scanner_func_pose[:3]) < eps) and dis > 0 and dis < 0.07
                   and self.is_left_gripper_close() and self.is_right_gripper_close())

        return qrcode_ok and scan_ok
