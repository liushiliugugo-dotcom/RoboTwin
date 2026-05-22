# RoboTwin 端到端流程：从数据采集到策略部署

以 ACT 策略 + beat_block_hammer 任务为例，说明完整的数据流。

---

## 整体架构

```
任务定义 (envs/*.py)
    ↓
配置加载 (task_config/*.yml)
    ↓
第一阶段：数据采集 (collect_data.sh)
    ├── 1A：种子搜索 → 找到能成功的随机种子
    └── 1B：数据重放 → 用成功种子保存观测数据到 HDF5
    ↓
第二阶段：数据处理 (policy/ACT/process_data.py)
    → 解码 JPEG、调整大小、重组为 (obs, action) 对
    ↓
第三阶段：策略训练 (policy/ACT/train.sh)
    → CNN + Transformer 监督学习，图像 → 关节动作
    ↓
第四阶段：策略评估 (script/eval_policy.py)
    → 加载检查点，在新场景中推理，记录成功率
```

---

## 第一阶段：数据采集

**入口脚本**：`bash collect_data.sh beat_block_hammer demo_randomized 0`

### 阶段 1A：种子搜索

| 项目 | 内容 |
|------|------|
| **输入** | `envs/beat_block_hammer.py`（任务定义）+ `task_config/demo_randomized.yml`（配置） |
| **过程** | 对 seed=0,1,2,...,N：随机化物体位置/灯光/背景 → `env.setup_demo(seed=N)` → `env.play_once()` 执行硬编码专家演示 → 若 `plan_success` 且 `check_success()` 则保存种子 |
| **输出** | `data/beat_block_hammer/demo_randomized/seed.txt`（成功种子列表）+ `_traj_data/episode*.pkl`（关节轨迹） |

### 阶段 1B：数据重放

| 项目 | 内容 |
|------|------|
| **输入** | `seed.txt` + `_traj_data/episode*.pkl` |
| **过程** | 对每个成功种子：重建相同场景 → 加载预录关节轨迹 → 逐步重放并每 15 步采集观测（RGB 图像、关节位置、末端姿态、夹爪值）→ 帧缓存为 .pkl → 合并为 HDF5 + MP4 |
| **输出** | `data/beat_block_hammer/demo_randomized/data/episode*.hdf5`（每个约 1000 帧） |

### HDF5 文件结构

```
episodeN.hdf5
├── /observation/
│   ├── head_camera/rgb     → [T, bytes]  (JPEG 编码)
│   ├── left_camera/rgb     → [T, bytes]
│   └── right_camera/rgb    → [T, bytes]
├── /joint_action/
│   ├── left_arm            → [T, 6]  float32
│   ├── left_gripper        → [T, 1]  float32
│   ├── right_arm           → [T, 6]  float32
│   └── right_gripper       → [T, 1]  float32
└── /endpose/
    ├── left_endpose        → [T, 7]  (x,y,z,qw,qx,qy,qz)
    ├── left_gripper        → [T, 1]
    ├── right_endpose       → [T, 7]
    └── right_gripper       → [T, 1]
```

### 读取 HDF5 图像

```python
image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
```

---

## 第二阶段：数据处理

**入口**：`python policy/ACT/process_data.py beat_block_hammer demo_randomized 50`

| 项目 | 内容 |
|------|------|
| **输入** | `data/beat_block_hammer/demo_randomized/data/episode*.hdf5` |
| **过程** | 对每个 episode：解码 JPEG → 调整大小为 640×480 → 重组结构为 observation_t = (图像, qpos)，action_t = next_qpos → 保存 |
| **输出** | `policy/ACT/processed_data/sim-beat_block_hammer/demo_randomized-50/episode_*.hdf5` + 更新 `SIM_TASK_CONFIGS.json` |

### 数据结构变化

```
采集阶段：                          处理阶段：
frame0: {img, qpos}                obs[0] = {img0, qpos0}, action[0] = qpos1
frame1: {img, qpos}      →        obs[1] = {img1, qpos1}, action[1] = qpos2
frame2: {img, qpos}                obs[2] = {img2, qpos2}, action[2] = qpos3
...                                ...
```

---

## 第三阶段：策略训练

**入口**：`bash policy/ACT/train.sh beat_block_hammer demo_randomized 50 0 0`

### ACT 训练参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_epochs` | 6000 | 训练轮数 |
| `batch_size` | 8 | 批次大小 |
| `lr` | 1e-5 | 学习率 |
| `kl_weight` | 10 | CVAE KL 散度权重 |
| `chunk_size` | 50 | 预测的未来步数 |
| `hidden_dim` | 512 | Transformer 隐藏维度 |
| `state_dim` | 14 | 动作维度（6+1+6+1） |
| `save_freq` | 2000 | 每 N epoch 保存检查点 |

### 训练流程 (imitate_episodes.py)

```
对每个 epoch (共 6000)：
  → 加载一批 (observation, action) 对
  → CNN 骨干网络编码图像 (640×480×3 → 特征向量)
  → 与 qpos 拼接 → 输入 Transformer 编码器-解码器
  → 预测动作分块（未来 50 步的关节角度）
  → 损失 = KL 散度 (CVAE) + L1 (动作)
  → 反向传播
每 2000 epoch 保存检查点
```

| 项目 | 内容 |
|------|------|
| **输入** | `processed_data/sim-beat_block_hammer/demo_randomized-50/` |
| **输出** | `policy/ACT/act_ckpt/act-beat_block_hammer/demo_randomized-50/policy_epoch_N.ckpt` |

### 检查点文件内容

```
policy_epoch_6000.ckpt (PyTorch checkpoint)
├── model_state_dict      (CNN + Transformer 权重)
├── optimizer_state_dict
├── epoch
└── loss
```

---

## 第四阶段：策略评估

**入口**：`python script/eval_policy.py --config policy/ACT/deploy_policy.yml`

### 评估流程 (eval_policy.py)

```
对 100 个测试种子：
  1. env.setup_demo(seed=N, eval_mode=True)
     → 随机背景/灯光，杂乱桌面
  2. instruction = "pick up the hammer with left hand..."
     env.set_instruction(instruction)
  3. model = ACT(ckpt_path)  ← 加载训练好的 .ckpt
     reset_model(model)
  4. while 步数 < step_lim:
       observation = env.get_obs()
         → {head_cam: (240,320,3), qpos: [14], ...}
       actions = model.get_action(observation)
         → [[left_qpos, left_gripper, right_qpos, right_gripper]]
       for action in actions:
           env.take_action(action)  ← 调用 cuRobo IK + mplib TOPP
       if env.check_success():
           成功！记录到视频最后一帧
           break
  5. 记录：成功 / 失败
```

| 项目 | 内容 |
|------|------|
| **输入** | `act_ckpt/act-beat_block_hammer/demo_randomized-50/policy_epoch_6000.ckpt` + `deploy_policy.yml` |
| **输出** | `eval_result/beat_block_hammer/ACT/demo_randomized/demo_randomized-50/<timestamp>/episode*.mp4` + `_result.txt` |

---

## 输入输出速查表

| 阶段 | 输入 | 输出 | 耗时 |
|------|------|------|------|
| **1A. 种子搜索** | 任务 .py + 配置 .yml | `seed.txt`, `_traj_data/*.pkl` | ~1 分钟/ep |
| **1B. 数据重放** | seed.txt + traj_data | `data/*.hdf5` + `.mp4` | ~30 秒/ep |
| **2. 处理数据** | 原始 HDF5 文件 | `processed_data/*.hdf5` | ~30 秒 |
| **3. 训练** | 处理后的 HDF5 文件 | `.ckpt` 文件 | 数小时 |
| **4. 评估** | .ckpt + 任务配置 | 视频 + 成功率 | ~5 分钟 |

---

## cuRobo 和 mplib 的角色

| 组件 | 硬件 | 作用 | 使用阶段 |
|------|------|------|----------|
| **cuRobo** | GPU (CUDA) | 基于梯度的运动规划 / IK 求解 | 阶段 1A（规划专家演示）+ 阶段 4（`action_type='ee'`） |
| **mplib** | CPU | RRT 规划 + 时间最优轨迹重参数化 (TOPP) | 阶段 4（`action_type='qpos'` 时平滑关节轨迹） |

训练（阶段 3）不使用 cuRobo 或 mplib——它是纯粹的监督学习，从图像直接预测关节动作。

---

## 策略接口约定

每个策略模块必须导出三个函数供 `eval_policy.py` 调用：

```python
def get_model(usr_args) -> model:
    """加载模型，返回模型实例"""

def eval(TASK_ENV, model, observation) -> observation:
    """运行推理，在环境中执行动作，返回新观测"""

def reset_model(model):
    """重置策略隐藏状态（新 episode 开始时调用）"""
```

## 相关文件路径

| 用途 | 路径 |
|------|------|
| 基类（核心 API） | `envs/_base_task.py` |
| 全局配置/路径 | `envs/_GLOBAL_CONFIGS.py` |
| 机器人 + 规划器 | `envs/robot/robot.py`, `envs/robot/planner.py` |
| 动作类型 | `envs/utils/action.py` |
| 数据采集入口 | `script/collect_data.py` |
| 策略评估 | `script/eval_policy.py` |
| 策略模型服务器 | `script/policy_model_server.py` |
| 本体配置 | `task_config/_embodiment_config.yml` |
| 摄像机配置 | `task_config/_camera_config.yml` |
| ACT 数据处理 | `policy/ACT/process_data.py` |
| ACT 训练 | `policy/ACT/imitate_episodes.py` |
| ACT 部署 | `policy/ACT/deploy_policy.py` |
