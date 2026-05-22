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

---

## 常见问题与深入理解

### 1. 大白话理解整个流程

整个 RoboTwin 做的事情，用一句话概括：**让机器人看视频学动作**。

具体来说分四步：

- **第一步（数据采集）**：先让专家（硬编码规则 + cuRobo 规划器）把任务做一遍，同时录屏 + 记录关节角度。跑 N 个随机种子，筛选出成功的那些，把它们的"视频帧 + 关节角度"存成 HDF5 文件。这就是"教科书"。
- **第二步（数据处理）**：把 HDF5 里的 JPEG 图片解码、缩放，把数据整理成 (当前画面, 当前关节角) → (下一个关节角) 的配对。这就是"题目和答案"。
- **第三步（训练）**：拿这些配对去训练一个神经网络（CNN 看图像 + Transformer 理解时序）。输入：图像 + 当前关节角，输出：未来 50 步的关节角。损失函数就是让预测值和真实值尽量接近。这就是"刷题"。
- **第四步（评估）**：把训练好的模型放到新的随机场景里，不给任何规则，只看图像，让它自己输出关节角去执行。统计成功率。这就是"考试"。

一句话：**专家演示 → 录数据 → 监督学习 → 闭卷考试**。

### 2. 为什么数据采集要"跑两遍"？

数据采集分 1A（种子搜索）和 1B（数据重放），而不是一遍跑完。原因：

- **规划太慢**：cuRobo 做一次 IK + 运动规划需要几百毫秒到几秒，如果每 15 帧采集一次观测的同时还要等规划结果，会非常慢。
- **保证一致性**：种子搜索阶段只记录"哪些种子能成功"和"关节轨迹的可复现片段（PKL）"。重放阶段直接用保存的关节轨迹驱动仿真（不再调 cuRobo），速度快且每帧的观测和动作完全对应。
- **挑选好的示范**：只有成功的 episode 才进入重放，自动过滤掉失败案例。

本质上是 **"先挑好学生，再抄他们的作业"**。

### 3. PKL 与 HDF5 的关系

| | PKL | HDF5 |
|------|------|------|
| **内容** | 关节轨迹段（位置+速度），由 cuRobo 规划产生 | 每帧观测（RGB图像、关节角、末端姿态、夹爪值） |
| **来源** | cuRobo 规划器输出 | 仿真器每 15 帧采集一次 |
| **作用** | 重放时复现同一段动作 | 训练数据 |
| **粒度** | 一个 episode 一个文件，N 个轨迹段 | 一个 episode 一个文件，约 1000 帧 |
| **可视化** | `python script/inspect_data.py pkl <file>` | `python script/inspect_data.py hdf5 <file>` |

关系：种子搜索 → PKL（关节轨迹）→ 重放时驱动仿真 → 采集观测 → HDF5。

PKL 是"乐谱"，HDF5 是"演奏录音+录像"。

### 4. 数据处理为什么要"错一帧"？

原始数据是每帧独立的：frame0={img0, qpos0}, frame1={img1, qpos1}, ...

处理后变成：obs[t] = {img_t, qpos_t}, action[t] = qpos_{t+1}

**为什么？** 因为训练目标是：看到当前画面和姿态，预测**下一步**的姿态。如果不移一帧，模型就会学到"看到什么就输出什么"（恒等映射），无法学会"看到 A 应该变成 B"。

```
采集：frame0, frame1, frame2, frame3, ...
处理：obs0=frame0, action0=qpos1
      obs1=frame1, action1=qpos2
      obs2=frame2, action2=qpos3
```

本质上是教模型：**"看到这个 → 变成那个"**。

### 5. qpos 与 ee 两种动作模式

策略输出的动作有两种类型：

| 模式 | 输出 | 执行方式 | 优点 | 缺点 |
|------|------|------|------|------|
| **qpos** | 关节角度 [14维] | mplib TOPP 平滑 → 直接驱动关节 | 简单直接，训练容易 | 对机器人本体变化敏感 |
| **ee** | 末端位姿 [14维: 左右手各 xyz+qwqxqyqz] | cuRobo IK 求解 → 关节角 | 跨机器人泛化好 | 依赖 cuRobo，多一步 IK |

- 当前 ACT 训练默认使用的是 **qpos** 模式（从 HDF5 的 `/joint_action/` 读取）。
- 如果要用 ee 模式，需要在数据处理和训练时改用 `/endpose/` 数据，并在评估配置中设置 `action_type: ee`。

直观理解：qpos 是"每个关节转多少度"，ee 是"手要放到什么位置"。

### 6. checkpoint 文件内部结构

训练的 .ckpt 文件不是只存了模型权重，它是一个完整的 PyTorch 检查点：

```
policy_epoch_6000.ckpt
├── model_state_dict      # CNN (ResNet-18) + Transformer 编码器-解码器的所有权重矩阵
├── optimizer_state_dict  # Adam 优化器的动量、方差等状态
├── epoch                 # 当前 epoch 数（用于断点续训）
├── loss                  # 最后一个 epoch 的 loss 值
└── (args)                # 训练时的超参数（可能嵌入在模型定义中）
```

评估时只需要 `model_state_dict`，其余是用于恢复训练的。

### 7. ACT 训练循环详解

```
对每个 epoch（默认 6000 轮）：
  for batch in dataloader:
    images, qpos, actions = batch
    # images: [B, T, 3, H, W] 一批图像序列
    # qpos:   [B, T, 14]      当前关节角
    # actions:[B, T, 14]      目标关节角（错一帧后的）

    # 1. CNN 编码每张图像 → 特征向量
    img_features = resnet18(images)  # [B, T, 512]

    # 2. 拼接图像特征 + qpos → Transformer 输入
    input_tokens = concat(img_features, qpos)  # [B, T, 512+14]

    # 3. CVAE：编码器从 (input, ground_truth_action) 学习隐变量分布
    #    解码器从隐变量预测动作序列
    mu, logvar = encoder(input_tokens, actions)       # 隐变量分布
    latent = sample(mu, logvar)                       # 采样
    predicted_actions = decoder(input_tokens, latent) # [B, T_future, 14]

    # 4. 损失函数
    l1_loss = |predicted_actions - actions|           # 动作预测要准
    kl_loss = KL(N(mu, var) || N(0,1))                # 隐变量不要太发散
    loss = l1_loss + 10 * kl_loss

    # 5. 反向传播
    loss.backward()
    optimizer.step()

每 2000 epoch → 保存 ckpt
```

核心是 **CVAE（条件变分自编码器）**：编码器在训练时"偷看"正确答案来学习隐变量，解码器根据图像+隐变量预测动作。推理时编码器不能用（没有正确答案），所以直接从标准正态分布采样隐变量。

### 8. 专家演示的本质：分层架构

专家演示并非单纯的 cuRobo 规划，而是**三层分工**：

```
你的工作（play_once）                 cuRobo 的工作              mplib 的工作
─────────────────                    ────────────              ────────────
"抓锤子"                             收到末端姿态 (x,y,z,q,w,x,y,z)  收到关节路径
  ↓ 选择 contact_point                  ↓ GPU IK 求解              ↓ TOPP 平滑
  ↓ 计算 pre_grasp 位姿                 → 关节轨迹 [T,6]            → 带速度的关节轨迹
  ↓ 调用 grasp_actor()
"提起 7cm"
  ↓ 调用 move_by_displacement(z=0.07)
"放到方块上方"
  ↓ 读取 functional_point
  ↓ 调用 place_actor()
```

cuRobo 只负责"怎么到达某个位姿"，**"该去哪里、做什么"** 由 `play_once()` 中的硬编码逻辑决定。

创建新任务时需要你设定的：
- **场景物体** — `load_actors()` 中放置 3D 模型（contact_points 在模型文件中预标注）
- **禁止区域** — `add_prohibit_area()` 让 cuRobo 规划时避开
- **动作序列** — `play_once()` 中调用 `grasp_actor` / `place_actor` / `move_by_displacement`

cuRobo 本身无需你配置——它从机器人模型的 `curobo.yml` 读取运动学参数。

### 9. 训练阶段的输入边界

训练只使用 HDF5 文件，完全不接触仿真环境：

```
───────────── 数据采集阶段 ────────────    ───── 训练阶段 ─────
cuRobo (运动规划)                           只用 HDF5
play_once() (硬编码逻辑)                      不接触仿真器
contact_points (抓取点标注)                  不接触规划器
物理仿真 (SAPIEN)                             纯粹矩阵运算
禁止区域 (prohibited_area)                   不理解物理约束

输入:  图像 + 当前 qpos  (来自 HDF5)
输出:  预测的关节角序列  (对比 HDF5 里的真实值)
损失:  L1(|预测 - 真实|)  → 越小越好
```

这意味着策略**不知道什么是"危险"**。它只学会了"看到类似画面时输出类似数字"：

```
专家演示: 锤子在 (0, 0.1) → cuRobo 规划安全路径 → 关节轨迹 A
新场景:   锤子在 (0, 0.3) → 策略直接输出关节轨迹 A + 偏移量
                           └─ 没有 cuRobo 验证是否会撞桌子
```

因此：
- **域随机化至关重要** — 单一背景/灯光下训练的策略换环境就会失败
- **部分策略（如 DP3）使用点云代替 RGB** — 对空间位置更敏感，泛化更好
- **评估阶段 cuRobo 仍会介入** — 如果策略输出的关节角到不了，TOPP/IK 会失败
- **约束并非没有** — 而是以"训练数据中的统计规律"形式隐式存在

---

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
