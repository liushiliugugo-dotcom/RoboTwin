# 新建任务实战记录：从螺丝刀到电子秤

## 第一次尝试：螺丝刀对准方块（失败）

### 任务设计

- 物体 `032_screwdriver`，目标 `create_box` 蓝色方块
- 单臂抓取螺丝刀手柄 → 提起 → 刀尖对准方块顶面

### 遇到的错误

| 错误 | 原因 | 尝试的修复 |
|------|------|------|
| `'Robot' object has no attribute 'left_planner'` | embodiment 只写了一个 `piper`，但框架强制双臂 | 改为 `[piper, piper, 0.6]` |
| `curobo_left.yml` not found | piper 只有 `curobo.yml`，双臂模式下找不到 `_left` 版本 | 换为 `aloha-agilex` |
| `Objects is unstable: 032_screwdriver` | 螺丝刀是圆柱体，物理模拟中无论如何都会在桌面滚动 | 固定姿态 `qpos=[0.707,0,0,0.707]`、提高 zlim → 仍然失败 |
| `target_pose cannot be None` | 螺丝刀随机姿态导致 cuRobo IK 找不到可达抓取位姿 | 换 aloha-agilex 提升可达性 → 仍有部分种子失败 |

### 失败原因

螺丝刀模型 `032_screwdriver` 在 `model_data0.json` 中标记 `"stable": false`，其圆柱形几何体在平坦桌面上的物理模拟中天然不稳定。项目中**没有运行时改变物体物理类型的机制**（静态→动态），稳定性检查无法绕过。

### 教训

- 新建任务前先检查物体 `model_data.json` 中的 `"stable"` 字段
- 圆柱体/球体类物体很难通过稳定性检查
- 优选方形、扁平物体

---

## 第二次尝试：物体称重（成功）

### 任务设计

| 项目 | 内容 |
|------|------|
| **物体池** | `077_phone`（手机）、`080_pillbottle`（药瓶）、`079_remotecontrol`（遥控器）随机三选一 |
| **目标** | `072_electronicscale`（电子秤），放在物体对侧 |
| **动作** | 单臂抓取物体 → 提起 15cm → 放到秤面 |
| **成功条件** | 物体在秤面中心 3.5cm 内 + 高于秤面 + 夹爪已松开 |

所有物体均标记 `"stable": true`。

### 文件清单

| 文件 | 用途 |
|------|------|
| `envs/weigh_object.py` | 任务类，继承 `Base_Task` |
| `task_config/weigh_object.yml` | 运行配置 |

### 任务代码结构

```python
class weigh_object(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)     # 固定写法

    def load_actors(self):
        # 1. 随机选择物体（三选一 + 随机 model_id）
        # 2. 物体放在桌子一侧
        # 3. 电子秤放在对侧（距离 > 15cm）
        # 4. 添加禁止区域

    def play_once(self):
        # 1. 根据物体位置选择手臂
        # 2. grasp_actor → move_by_displacement → place_actor
        # 3. 记录 info

    def check_success(self):
        # 物体 XY 距秤面中心 < 3.5cm + Z 高于秤面 + 夹爪松开
```

### 运行命令

```bash
bash collect_data.sh weigh_object weigh_object 0
```

### 关键设计决策

1. **随机物体池**：参考了 `place_object_scale.py` 的模式，使用 `glob` 动态获取可用 `model_id`
2. **物体与秤分离**：物体放在桌子一侧（`xlim` 随机），秤强制放在对侧，保证有足够的运动空间
3. **constrain="free"**：放置时使用自由约束，让 cuRobo 自行规划安全的放置轨迹
4. **稳定性验证**：创建前确认 `model_data.json` 中 `"stable": true`

---

## 新建任务检查清单

- [ ] 物体的 `model_data.json` 中 `"stable"` 不为 `false`
- [ ] 物体有 `contact_points`（可被抓取）
- [ ] 目标物体有 `functional_points`（有明确的放置/作用位置）
- [ ] `embodiment` 使用 `aloha-agilex`（双臂标配，可达性好）或 `[piper, piper, 0.6]`
- [ ] `episode_num` 先设为 5 测试，确认可用后再增加
- [ ] `clear_cache_freq: 1` 适配 6GB 显存
- [ ] 关闭 `pointcloud`、`depth`、`wrist_camera` 节省显存
- [ ] 任务文件名 = 类名（`weigh_object.py` → `class weigh_object`）
- [ ] `task_config` 文件名与任务名一致
