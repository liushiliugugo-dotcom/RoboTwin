# 新任务场景搭建指南

## 最小模板

一个任务就是一个 Python 类，放在 `envs/` 下，继承 `Base_Task`，只需实现 **4 个方法**：

```python
from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *

class my_task(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)     # 固定写法

    def load_actors(self):
        # 在这里加载物体
        ...

    def play_once(self):
        # 在这里写专家演示逻辑
        ...
        return self.info

    def check_success(self):
        # 判断任务是否成功
        return True 或 False
```

其他一切（桌子、机器人、相机、数据记录）都由基类 `_init_task_env_()` 自动处理。

---

## 可用资源

| 资源类型 | 位置 | 数量 |
|------|------|------|
| 机器人 | `assets/embodiments/` | 5 种：aloha-agilex, piper, franka-panda, ARX-X5, ur5-wsg |
| 物体 | `assets/objects/` | 129 个，编号 001~129 |

---

## 加载物体的三种方式

```python
# 1. 加载预制物体（瓶子、锤子、碗...）
obj = create_actor(
    scene=self,
    pose=sapien.Pose([x, y, z], [qw, qx, qy, qz]),
    modelname="001_bottle",     # assets/objects/ 下的目录名
    model_id=0,                 # 对应 model_data{id}.json，控制缩放/变体
    convex=True,                # True=用凸碰撞体(速度快)
)
obj.set_mass(0.001)             # 设为轻质量，模拟无重力

# 2. 创建几何体（方块等）
box = create_box(
    scene=self,
    pose=sapien.Pose([x, y, z]),
    half_size=(0.025, 0.025, 0.025),
    color=(1, 0, 0),
    name="target_box",
)

# 3. 加载关节物体（抽屉、笔记本电脑...）
laptop = create_urdf_obj(
    scene=self,
    pose=sapien.Pose([x, y, z]),
    modelname="015_laptop",
)
```

---

## 物体的关键属性：contact_points 与 functional_points

每个物体目录下的 `points_info.json` 定义了两种关键点：

| 点类型 | 含义 | 用法 |
|------|------|------|
| **contact_points** | 抓取点 — 夹爪该夹哪里 | `grasp_actor(..., contact_point_id=N)` |
| **functional_points** | 功能点 — 物体"作用部位"在哪里 | `obj.get_functional_point(N, "pose")` |

例如 `020_hammer/points_info.json`：
- contact_point 0 = 锤子手柄（夹这里）
- functional_point 0 = 锤头（敲这里）

---

## 专家演示 `play_once()` 常用动作 API

```python
def play_once(self):
    # 决定用哪只手（根据物体位置分左右工区）
    arm_tag = ArmTag("left" if obj_x < 0 else "right")

    # 1. 抓取物体
    self.move(self.grasp_actor(
        self.obj,
        arm_tag=arm_tag,
        pre_grasp_dis=0.12,       # 抓取前离物体多远
        grasp_dis=0.01,           # 抓取深度（向接触点推进多少）
        contact_point_id=0,       # 用哪个接触点
    ))

    # 2. 提起
    self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

    # 3. 放置到目标上方
    self.move(self.place_actor(
        self.obj,
        target_pose=self.target.get_functional_point(0, "pose"),
        arm_tag=arm_tag,
        functional_point_id=0,    # obj 的哪个功能点对准目标
        pre_dis=0.06,             # 放置前离目标多远
        dis=0,                    # 最终偏差
    ))

    return self.info
```

### 动作 API 汇总

| 方法 | 作用 |
|------|------|
| `grasp_actor(obj, arm_tag, ...)` | 抓取物体（自动调 cuRobo IK） |
| `place_actor(obj, target_pose, arm_tag, ...)` | 将物体放到目标位姿 |
| `move_by_displacement(arm_tag, x/y/z, move_axis)` | 沿轴平移 |
| `move_to_pose(arm_tag, pose)` | 直接移动机械臂到指定位姿 |

---

## 运行配置

创建 `task_config/my_task.yml`：

```yaml
render_freq: 20
episode_num: 3
use_seed: false
save_freq: 15
embodiment:
  - piper                     # 或 aloha-agilex
domain_randomization:
  random_background: true
  cluttered_table: false
  random_light: false
camera:
  collect_head_camera: true
  collect_wrist_camera: false
data_type:
  rgb: true
  endpose: true
  qpos: true
  pointcloud: false
clear_cache_freq: 1
collect_data: true
```

运行：`bash collect_data.sh my_task my_task 0`

---

## 参考示例

| 任务文件 | 复杂度 | 适合学什么 |
|------|------|------|
| `envs/grab_roller.py` | 简单 | 单一抓取 |
| `envs/lift_pot.py` | 简单 | 双臂协同夹取 |
| `envs/beat_block_hammer.py` | 中等 | 抓取→移动→放置的完整流程 |
| `envs/place_can_basket.py` | 中等 | check_success 用接触判断 |
| `envs/open_laptop.py` | 中等 | 使用 create_urdf_obj 关节物体 |
