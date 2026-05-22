from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *


class align_screwdriver(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Screwdriver: fixed orientation lying flat, random XY position only
        sd_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.15, 0.15],
            zlim=[0.785],
            qpos=[0.707, 0, 0, 0.707],
            rotate_rand=False,
            rotate_lim=[0, 0, 0],
        )
        self.screwdriver = create_actor(
            scene=self,
            pose=sd_pose,
            modelname="032_screwdriver",
            convex=True,
            model_id=0,
        )
        self.screwdriver.set_mass(0.001)

        # Target block at a random position, ensure it's not too close to screwdriver
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.15, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.3],
        )
        while np.linalg.norm(block_pose.p[:2] - sd_pose.p[:2]) < 0.1:
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.15, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.3],
            )

        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.02, 0.02, 0.02),
            color=(0.2, 0.6, 1.0),
            name="target_block",
            is_static=True,
        )

        self.add_prohibit_area(self.screwdriver, padding=0.10)

    def play_once(self):
        sd_pose = self.screwdriver.get_pose().p
        arm_tag = ArmTag("left" if sd_pose[0] < 0 else "right")

        # Grasp screwdriver by handle
        self.move(self.grasp_actor(
            self.screwdriver,
            arm_tag=arm_tag,
            pre_grasp_dis=0.12,
            grasp_dis=0.01,
            contact_point_id=0,
        ))

        # Lift it up
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        # Align screwdriver tip to block top surface
        self.move(self.place_actor(
            self.screwdriver,
            target_pose=self.block.get_functional_point(0, "pose"),
            arm_tag=arm_tag,
            functional_point_id=0,
            pre_dis=0.06,
            dis=0,
            is_open=False,
        ))

        self.info["info"] = {"{A}": "032_screwdriver/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        tip_pose = self.screwdriver.get_functional_point(0, "pose").p
        block_center = self.block.get_functional_point(0, "pose").p
        eps = np.array([0.03, 0.03, 0.04])
        return np.all(abs(tip_pose - block_center) < eps)
