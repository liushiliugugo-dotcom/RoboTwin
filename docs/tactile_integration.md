# 触觉集成分析与规划

## 当前状态

项目**没有触觉相关内容**。虽然 SAPIEN 物理引擎自带接触检测 API，`_base_task.py` 也封装了两个方法，但它们只用于**任务成功判断**，从未作为传感器模态接入数据采集和训练流水线。

### 已有基础设施

| 方法 | 位置 | 实际用途 |
|------|------|------|
| `check_actors_contact(a1, a2)` | `envs/_base_task.py:982` | 任务成功判断，如"罐子有没有碰到篮子" |
| `get_gripper_actor_contact_position(name)` | `envs/_base_task.py:970` | 任务成功判断，如"夹爪有没有碰到订书机" |
| `get_scene_contact()` | `envs/_base_task.py:996` | 调试用，打印全部接触对 |
| `self.robot.gripper_name` | `envs/robot/robot.py:190` | 夹爪碰撞体名称列表 |

SAPIEN contact 对象提供：`bodies[]`（接触双方）、`points[]`（每个接触点含 `.position`）、力、法向量等。

### 缺失的部分

- `data_type` 配置中没有 `tactile` 选项
- `get_obs()` 不采集任何接触数据
- HDF5 中没有触觉相关的 group
- 没有任何策略模型接收触觉输入
- `process_data.py` 不处理触觉数据

---

## 修改路线图

### Level 1：接触力/接触状态作为观测（~50 行改动）

**A) 配置文件** — `task_config/xxx.yml`
```yaml
data_type:
  tactile: true   # 新增
```

**B) 数据采集** — `envs/_base_task.py` 的 `get_obs()`（L440 附近）
```python
if self.data_type.get("tactile", False):
    contacts = self.scene.get_contacts()
    pkl_dic["tactile"] = {
        "left_contact": [],    # 左手每个接触点
        "right_contact": [],   # 右手每个接触点
    }
    for contact in contacts:
        for i in [0, 1]:
            if contact.bodies[i].entity.name in self.robot.gripper_name:
                side = "left" if "left" in contact.bodies[i].entity.name else "right"
                for point in contact.points:
                    pkl_dic["tactile"][f"{side}_contact"].append({
                        "position": point.position,
                    })
```

**C) 策略模型** — 在 CNN 特征后拼接触觉编码
```python
if tactile is not None:
    tactile_features = tactile_encoder(tactile)  # 小型 MLP
    features = torch.cat([img_features, qpos, tactile_features], dim=-1)
```

### Level 2：指尖触觉阵列仿真（~200 行改动）

仿真阵列式触觉传感器（如 GelSight 式的 N×M 力分布网格）：

| 文件 | 改动 |
|------|------|
| `assets/embodiments/<robot>/` | 在夹爪手指模型上加标记点或碰撞体 |
| `envs/_base_task.py` | 在 `get_obs()` 中增加阵列采样逻辑 |
| `envs/robot/robot.py` | `gripper_name` 需包含传感器碰撞体名称 |
| `task_config/xxx.yml` | `tactile_resolution: [4, 4]` |
| `policy/ACT/process_data.py` | 处理触觉数据（归一化、reshape） |
| `policy/ACT/act_policy.py` | 增加触觉编码器分支 |

### Level 3：外接触觉传感器（真实机器人）

- 传感器数据通过 ROS/direct SDK 读取
- 在 `script/collect_data.py` 的 real-robot 分支中采集
- HDF5 结构与仿真保持一致（方便 sim2real 迁移）

---

## 关键设计决策

1. **触觉表示什么**？力的大小？接触位置？法向量？滑动检测？
2. **触觉和视觉的关系**？独立模态并行编码后融合，还是仅作为抓取阶段的辅助信号？
3. **SAPIEN contact API 精度**？碰撞检测精度取决于碰撞网格分辨率，细粒度触觉仿真可能不够
