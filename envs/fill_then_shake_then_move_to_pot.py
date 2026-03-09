from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import numpy as np
from copy import deepcopy


class fill_then_shake_then_move_to_pot(Base_Task):
    """
    Stage 1: Move sauce can next to the kitchen pot (move_can_pot).
    Stage 2: Shake the bottle (shake_bottle).
    Stage 3: Lift the kitchen pot with both arms (lift_pot).
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # --- Pot: center of table (shared by all stages) ---
        # Use model_id in [0,1] to satisfy both move_can_pot (0-6) and lift_pot (0-1)
        self.pot_id = np.random.randint(0, 2)
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="060_kitchenpot",
            modelid=self.pot_id,
            xlim=[0.0, 0.0],
            ylim=[0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0, 0, 0, 1],
        )

        # --- Stage 1: Sauce can (from move_can_pot) ---
        pot_pose = self.pot.get_pose()
        can_pos = rand_pose(
            xlim=[-0.3, 0.3],
            ylim=[0.05, 0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        while abs(can_pos.p[0]) < 0.2 or (((pot_pose.p[0] - can_pos.p[0]) ** 2 +
                                            (pot_pose.p[1] - can_pos.p[1]) ** 2) < 0.09):
            can_pos = rand_pose(
                xlim=[-0.3, 0.3],
                ylim=[0.05, 0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
        id_list = [0, 2, 4, 5, 6]
        self.can_id = np.random.choice(id_list)
        self.can = create_actor(
            scene=self,
            pose=can_pos,
            modelname="105_sauce-can",
            convex=True,
            model_id=self.can_id,
        )
        self.can_arm_tag = ArmTag("right" if self.can.get_pose().p[0] > 0 else "left")
        self.can_target_pose = sapien.Pose(
            [
                pot_pose.p[0] - 0.18 if self.can_arm_tag == "left" else pot_pose.p[0] + 0.18,
                pot_pose.p[1],
                0.741 + self.table_z_bias,
            ],
            pot_pose.q,
        )

        # --- Stage 2: Bottle (from shake_bottle) ---
        bottle_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.15, -0.05],
            zlim=[0.785],
            qpos=[0, 0, 1, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
        )
        while abs(bottle_pos.p[0]) < 0.1:
            bottle_pos = rand_pose(
                xlim=[-0.15, 0.15],
                ylim=[-0.15, -0.05],
                zlim=[0.785],
                qpos=[0, 0, 1, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 4],
            )
        self.bottle_id = np.random.choice([i for i in range(20)])
        self.bottle = create_actor(
            scene=self,
            pose=bottle_pos,
            modelname="001_bottle",
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.01)

        # Prohibit areas
        self.add_prohibit_area(self.pot, padding=0.03)
        self.add_prohibit_area(self.can, padding=0.1)
        self.add_prohibit_area(self.bottle, padding=0.05)

        pot_x, pot_y = pot_pose.p[0], pot_pose.p[1]
        if self.can_arm_tag == "left":
            self.prohibited_area.append([pot_x - 0.15, pot_y - 0.1, pot_x, pot_y + 0.1])
        else:
            self.prohibited_area.append([pot_x, pot_y - 0.1, pot_x + 0.15, pot_y + 0.1])

    def play_once(self):
        # === Stage 1: Move can next to pot ===
        arm_tag = self.can_arm_tag
        self.move(self.grasp_actor(self.can, arm_tag=arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag, y=-0.1, z=0.1))
        self.move(self.place_actor(self.can, target_pose=self.can_target_pose, arm_tag=arm_tag,
                                   pre_dis=0.05, dis=0.0))
        self.move(self.back_to_origin(arm_tag))

        # === Stage 2: Shake bottle ===
        bottle_arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.bottle, arm_tag=bottle_arm_tag, pre_grasp_dis=0.1))

        target_quat = [0.707, 0, 0, 0.707]
        self.move(self.move_by_displacement(arm_tag=bottle_arm_tag, z=0.1, quat=target_quat))

        quat1 = deepcopy(target_quat)
        quat2 = deepcopy(target_quat)
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 8) * 7, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat1)
        quat1 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]
        y_rotation = t3d.euler.euler2quat(0, -7 * (np.pi / 8), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat2)
        quat2 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=bottle_arm_tag, z=0.05, quat=quat1))
            self.move(self.move_by_displacement(arm_tag=bottle_arm_tag, z=-0.05, quat=quat2))

        self.move(self.move_by_displacement(arm_tag=bottle_arm_tag, quat=target_quat))
        self.move(self.back_to_origin(bottle_arm_tag))

        # === Stage 3: Lift pot with both arms ===
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        self.move(
            self.close_gripper(left_arm_tag, pos=0.5),
            self.close_gripper(right_arm_tag, pos=0.5),
        )
        self.move(
            self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
        )
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
            self.move_by_displacement(right_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
        )

        self.info["info"] = {
            "{A}": f"105_sauce-can/base{self.can_id}",
            "{B}": f"060_kitchenpot/base{self.pot_id}",
            "{C}": f"001_bottle/base{self.bottle_id}",
        }
        return self.info

    def check_success(self):
        pot_pose = self.pot.get_pose()
        left_end = np.array(self.robot.get_left_tcp_pose()[:3])
        right_end = np.array(self.robot.get_right_tcp_pose()[:3])
        left_grasp = np.array(self.pot.get_contact_point(0)[:3])
        right_grasp = np.array(self.pot.get_contact_point(1)[:3])
        pot_dir = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])
        return (pot_pose.p[2] > 0.82 and np.sqrt(np.sum((left_end - left_grasp) ** 2)) < 0.03
                and np.sqrt(np.sum((right_end - right_grasp) ** 2)) < 0.03 and pot_dir > 0.8)
