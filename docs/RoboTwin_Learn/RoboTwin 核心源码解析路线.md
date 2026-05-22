机械臂相关的配置文件究竟在哪？
机械臂的配置分为两层：逻辑调度层和物理模型层。

物理模型层 (底层资产): assets/embodiments/
这里存放了机器人的 3D 模型（URDF/Mesh文件）以及 CuRobo 运动学规划器的底层配置。

路径示例：assets/embodiments/piper/ 或 assets/embodiments/aloha-agilex/

核心文件：curobo.yml / curobo_left.yml。这里面定义了机器人的关节数量、连杆长度、碰撞体积等极其硬核的参数。（通常不需要你修改，除非你要自己接入一款全新的机器人型号）

逻辑调度层 (用户配置): task_config/_embodiment_config.yml
这个文件告诉系统，当你在 YAML 里输入 piper 时，系统应该去哪里找它的物理模型，以及它的末端执行器（夹爪）长什么样。



完整实验流程中，究竟调用了哪些主要文件？
当你敲下 bash collect_data.sh grab_roller my_first_piper_test 0 时，系统就像多米诺骨牌一样，按顺序触发了以下核心文件：

启动入口：collect_data.sh

作用： 这是一个外壳脚本，它帮你整理好环境变量，并把任务名、配置名传递给核心 Python 脚本。

大脑调度中心：script/collect_data.py

作用： 读取你的 my_first_piper_test.yml 配置文件，并根据传入的 grab_roller 任务名，去 envs/ 文件夹下寻找对应的任务逻辑。

任务剧本：envs/grab_roller.py (或其他任务文件)

作用： 专属于这个任务的规则文件。里面写了桌子上要放什么东西（滚筒、盘子）、初始位置在哪、以及判断任务成功的标准（比如滚筒是否被成功抓起）。

基座与渲染引擎：envs/_base_task.py

作用： 这是所有任务的“老父亲”。它负责调用 SAPIEN 引擎，生成光照、桌子、相机，并准备好让机器人登场。

机器人中枢：envs/robot/robot.py 和 envs/robot/planner.py

作用： 读取机器人的 CuRobo 配置文件，把机器人实体“生”在仿真世界里，并接管它的运动学规划（也就是之前报错卡住的地方）。




oboTwin 核心源码解析路线图
阶段一：入口与调度中心 (Entry & Dispatch)
目标： 了解系统是如何读取你的 YAML 配置，如何分配随机种子（Seed），以及如何调度整个采集循环的。

核心文件： script/collect_data.py

阶段二：世界基座与任务定义 (World Base & Task Logic)
目标： 探究 SAPIEN 物理引擎是如何被唤醒的，桌子、灯光和相机是如何建立的，以及具体的任务（如“抓取滚筒”）的胜利条件和初始化逻辑是什么。

核心文件： envs/_base_task.py (底层基座) 和 envs/grab_roller.py (具体任务逻辑)

阶段三：机器人本体与运动大脑 (Embodiment & Planner)
目标： 这是整个系统最硬核的部分！我们将分析机器人是如何加载 URDF 模型的，以及 CuRobo 是如何计算逆运动学（IK）并规划出一条避障轨迹的。

核心文件： envs/robot/robot.py (机器人控制封装) 和 envs/robot/planner.py (CuRobo 规划器接口)

阶段四：数据打包与输出 (Data Recording)
目标： 了解系统是如何在每一帧（Step）记录关节状态（qpos）、末端位姿（ee）和相机图像，并最终打包成可供深度学习模型训练的 .hdf5 和 .mp4 文件的。

核心文件： 数据保存相关的工具类（通常在 script/collect_data.py 的尾部或 envs/utils.py 中）




想做不同实验，我需要改哪些文件？
根据你的实验深度，你需要修改的地方分为两个段位：

🟢 初阶：只改环境参数和机器人（不改任务逻辑）
如果你只是想在官方已经写好的任务（如敲击方块、抓取滚筒）里，换个机器人、换个视角、或者让桌子变得杂乱，你完全不需要碰 Python 代码。

你需要改的文件： 你自己创建的 task_config/xxx.yml 文件。

怎么改： 修改里面的 embodiment（比如改成 [piper, piper, 0.6] 试试双臂组合），开启 cluttered_table: true（让桌子变乱），或者修改相机视角。然后在终端里直接调用这个新的 YAML 文件。

🔴 高阶：设计一个全新的、属于你自己的实验任务
如果你想让机器人做官方没有的任务（比如“把苹果切开”或者“把线缆插进插座”），你需要去动 Python 源码。

你需要改的文件： 在 envs/ 目录下创建一个新的 Python 文件（例如 envs/cut_apple.py）。

怎么改： 
1.  你需要在里面定义物体的 3D 模型路径。
2.  编写 play_once() 函数，手把手教机器人的各个关节该怎么动才能切开苹果（这也是可以用指南里提到的 Expert Code Generation (GPT 生成代码) 功能来辅助完成的地方）。
3.  编写 check_success() 函数，告诉系统切到什么程度算成功。





关于路径规划：
在你的 planner.py 中，有这样两块代码：
from curobo.wrap.reacher.motion_gen import MotionGen 和 import mplib。
这才是真正的规划引擎，你需要离开 RoboTwin，去它们的官方 Github 仓库看源码：

1. NVIDIA CuRobo (核心主力大脑)
代码库位置： https://github.com/NVlabs/curobo

看什么： 它的核心是用 CUDA 写的。如果你想看 Python 层面的算法编排，去看 curobo/src/curobo/rollout/ 和 curobo/src/curobo/trajopt/ 目录；如果想看硬核的 GPU 并行碰撞检测，需要看它底层编译的 CUDA 内核（.cu 文件）。

2. Mplib (备用传统大脑)
代码库位置： https://github.com/haosulab/MPlib

看什么： 这是基于著名的 OMPL（Open Motion Planning Library）封装的。真正的算法全是用 C++ 写的，你需要去看 src/planner/ompl/ 目录下的实现。



这两个库代表了目前机器人界最主流的两大算法流派，它们的工作方式完全不同：

流派 A：基于优化的规划 (Trajectory Optimization) —— CuRobo 的绝招
你可以把这种算法想象成“弹橡皮筋”。

粗略猜想 (Initial Guess)： 首先，算法不管三七二十一，在当前手臂位置和目标位置之间连一条直线，或者凭经验生成一条粗糙的轨迹。

定义代价 (Cost Function)： 算法制定一个极其严苛的打分系统：

距离目标越远，扣分（代价越大）。

动作太突兀（速度、加速度剧变），扣分。

只要碰到障碍物，直接扣 10000 分！

梯度下降 (Gradient Descent)： 接下来，CuRobo 利用强大的 GPU 并行算力，通过数学求导（求梯度）的方式，像一只无形的手一样，把那条初始的“橡皮筋”往低分（低代价）的方向拉扯。

结果： 橡皮筋被一点点“弹”开障碍物，最终被优化成一条既平滑又安全的绕行曲线。这也是为什么 CuRobo 跑出来的动作看起来特别像真实人类。

流派 B：基于采样的规划 (Sampling-based Planning) —— Mplib 的看家本领 (如 RRT 算法)
你可以把这种算法想象成“植物根系盲目生长”。Mplib 中最经典的算法叫 RRT (快速探索随机树)。

随机撒点 (Random Sampling)： 算法在整个 3D 空间（或电机的关节空间）里随机扔一个点。

找最近的树枝 (Nearest Neighbor)： 看看目前已经长出来的路径树里，哪个节点离这个随机点最近。

长出新枝 (Extend)： 从那个最近的节点，朝着这个随机点长出一小截树枝。

碰撞检测 (Collision Check)： 如果这一小截树枝撞到了桌子，就把这截树枝砍掉作废；如果没撞到，它就成为了树的新节点。

到达目标： 不断重复上述过程，直到某根树枝的末端进入了目标点的范围内。然后顺着这根树枝一路找回“根节点（起点）”，这就是最终的路径。