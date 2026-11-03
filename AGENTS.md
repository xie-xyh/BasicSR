# AGENTS.md — BasicSR 去雾二次开发指南

> 本文件面向 AI 编码助手与开发者，描述**当前仓库**（裁剪后的 BasicSR 1.4.2）的代码架构与开发规范。
> 原始参考文档：`BasicSR-Tutorial-v0.5.pdf`（官方 90 页中文解读，见文末章节对照表）。
> 与 PDF 的差异：本仓库已删除视频超分 / 人脸修复 / StyleGAN2 / Real-ESRGAN 退化管线等模块，只保留**单图像复原**（去雾可直接复用）的最小闭环。

---

## 1. 项目定位

- **用途**：单图像去雾（dehazing）的模型训练、测试与推理框架。
- **形态**：本地克隆开发，`basicsr` 以 editable 模式安装（`pip install -e .`），修改源码立即生效，无需重新安装。
- **已裁剪**：BasicVSR/EDVR 等视频模型、StyleGAN2/DFDNet/HiFaceGAN 人脸与生成模型、Real-ESRGAN 退化仿真、NIQE/FID 指标、CUDA 自定义算子（`basicsr/ops`）。这些内容可通过 git 历史恢复，但正常开发不需要。
- **保留的核心闭环**：`train.py`/`test.py` 入口 + Registry 动态实例化 + YAML 配置驱动。开发新去雾方法 = 写 arch / model（+ 可选 dataset/loss）+ 写 yml 配置。

## 2. 环境与常用命令

conda 环境名：`basicsr`（Python 3.10，PyTorch 2.1.1+cu118，GPU 版）。激活后 `basicsr` 包已可导入。

```bash
conda activate basicsr

# 训练（单卡；--debug 每 iter 打印、8 iter 后触发 validation，用于快速排错）
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml
python basicsr/train.py -opt <config>.yml --debug
python basicsr/train.py -opt <config>.yml --auto_resume        # 自动找最近 checkpoint 续训
python basicsr/train.py -opt <config>.yml --force_yml train:total_iter=10000 val:val_freq=500  # 命令行改配置

# 多卡分布式
bash scripts/dist_train.sh <num_gpus> <config>.yml

# 测试（输出指标 + 保存结果图）
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml
```

每个实验在 `experiments/<config里的name>/` 下生成：`models/`（`net_g_<iter>.pth`）、`training_states/`（`<iter>.state`）、`visualization/`（val 结果图）、log 文件与配置副本。

## 3. 代码架构

### 3.1 目录结构（当前裁剪后）

```
basicsr/
├── train.py                  # 训练入口：train_pipeline()
├── test.py                   # 测试入口：test_pipeline()
├── archs/                    # 网络结构
│   ├── arch_util.py          # 通用构件：default_init_weights, make_layer, pixel_unshuffle, trunc_normal_ 等
│   ├── srresnet_arch.py      # MSRResNet (SRGAN 的 Generator，去雾常用改造起点)
│   ├── rrdbnet_arch.py       # RRDBNet (ESRGAN 的 Generator)
│   ├── swinir_arch.py        # SwinIR (Transformer 骨干)
│   ├── deanet_arch.py        # DEANet 去雾网络（差分卷积 DEConv + CGA 注意力，WACV 2024）
│   ├── sdlora_dehaze_arch.py # SDLoRADehaze 单步扩散去雾（SD2.1-base 冻结主干 + UNet LoRA，
│   │                         # 有雾 latent 直接作为扩散起点，需 diffusers/peft 与本地 SD 权重）
│   ├── discriminator_arch.py # UNetDiscriminatorSN (GAN 训练用)
│   └── vgg_arch.py           # VGGFeatureExtractor (感知/对比损失用)
├── data/
│   ├── paired_image_dataset.py   # 成对数据集（hazy/GT 训练与测试）
│   ├── single_image_dataset.py   # 仅 LQ（无 GT 的推理/测试）
│   ├── transforms.py             # augment, paired_random_crop 等数据增强
│   ├── data_util.py / data_sampler.py / prefetch_dataloader.py
│   └── meta_info/                # DIV2K 的 meta 文件（大数据集加速用，可选）
├── models/
│   ├── base_model.py         # 模型基类：优化器/scheduler/EMA/保存/加载/resume
│   ├── sr_model.py           # SRModel —— 单图像复原模型的直接模板
│   ├── dehaze_model.py       # DehazeModel（= SRModel + 可选 CR 对比损失，negative=有雾输入）
│   ├── sdlora_model.py       # SDLoRAModel（= DehazeModel + mod-8 测试填充 + LoRA-only 检查点/EMA）
│   └── lr_scheduler.py       # CosineAnnealingRestartLR 等
├── losses/                   # basic_loss.py (L1/L2/MSE/Gradient/Perceptual), contrastive_loss.py (DEA-Net CR 损失), gan_loss.py, loss_util.py
├── metrics/                  # psnr_ssim.py (PSNR/SSIM), no_ref.py (FADE/NIQE/BRISQUE 无参考指标), metric_util.py
└── utils/                    # registry.py, logger.py, options.py, file_client.py,
                              # img_util.py, color_util.py, matlab_functions.py, dist_util.py 等
options/train/SRResNet_SRGAN/   # SR 训练配置模板（train_MSRResNet_x{2,3,4}.yml）
options/train/DEANet/           # DEA-Net 去雾训练配置（train_DEANet_mixed 混合 + dense/mist/moderate 分开）
options/train/SDLoRA/           # SDLoRA 单步扩散去雾训练配置（002 基线，Dense 子集 LoRA 适配）
options/test/SRResNet_SRGAN/    # SR 测试配置模板
options/test/DEANet/            # DEA-Net 去雾测试配置（按子集分别输出指标）
options/test/SDLoRA/            # SDLoRA 去雾测试配置（Dense 主测 + Mist/Moderate 泛化观察）
inference/                      # inference_esrgan.py / inference_swinir.py 单图推理范例
scripts/data_preparation/       # extract_subimages.py, create_lmdb.py, generate_meta_info.py,
                                # filter_min_size.py（按最小边长过滤成对数据集，生成软链接副本）
scripts/metrics/                # calculate_psnr_ssim.py, calculate_lpips.py
datasets/                       # 数据集存放（推荐软链接），见 datasets/README.md
research/                       # 研究日志（002_sdlora/RESEARCH_LOG.md 等）与临时实验配置
```

### 3.2 核心机制：Registry 动态实例化

框架通过 5 个注册表实现"配置写类名 → 自动实例化"（定义在 `basicsr/utils/registry.py`）：

| 注册表 | 用于 | 构建函数 | 文件后缀约定 |
|---|---|---|---|
| `DATASET_REGISTRY` | Dataset 类 | `build_dataset` (`basicsr/data/__init__.py`) | `_dataset.py` |
| `ARCH_REGISTRY` | 网络结构类 | `build_network` (`basicsr/archs/__init__.py`) | `_arch.py` |
| `MODEL_REGISTRY` | Model 类 | `build_model` (`basicsr/models/__init__.py`) | `_model.py` |
| `LOSS_REGISTRY` | Loss 类 | `build_loss` (`basicsr/losses/__init__.py`) | `_loss.py` |
| `METRIC_REGISTRY` | 指标**函数** | `calculate_metric` (`basicsr/metrics/__init__.py`) | 无后缀要求 |

关键事实（PDF §4.2）：

- **自动扫描**：`archs/data/models/losses` 的 `__init__.py` 会扫描对应目录下符合后缀的文件并自动 import。新建文件**无需修改任何 `__init__.py`**。
- **metrics 是例外**：指标注册后必须**显式 import** 到 `basicsr/metrics/__init__.py`（当前仅有 `from .psnr_ssim import calculate_psnr, calculate_ssim`）。
- 注册方式：类/函数定义前加装饰器，如 `@ARCH_REGISTRY.register()`。
- **类名/函数名不能重复**，否则启动即报错 `An object named xxx was already registered`。
- 查找失败时会回退找 `<name>_basicsr` 后缀的注册项（日志里 `Name xxx is not found, use name: xxx_basicsr!` 是正常提示，不是错误）。
- 普通工具文件**不要**以 `_arch.py`/`_model.py`/`_dataset.py`/`_loss.py` 结尾，避免被扫描器误 import。

### 3.3 训练流程（PDF §3.2）

```
python basicsr/train.py -opt xxx.yml
  └─ train_pipeline()
       ├─ parse_options()            # 解析 yml + 命令行参数 (basicsr/utils/options.py)
       ├─ init distributed / seed / resume state
       ├─ create_train_val_dataloader()   # build_dataset + build_dataloader（data/__init__.py）
       ├─ build_model(opt)                # MODEL_REGISTRY 实例化（models/__init__.py）
       │     └─ SRModel.__init__()
       │          ├─ build_network(opt['network_g'])   # ARCH_REGISTRY
       │          ├─ load pretrain (path.pretrain_network_g)
       │          └─ init_training_settings()
       │               ├─ EMA (train.ema_decay > 0 时)
       │               ├─ build_loss(train.pixel_opt / perceptual_opt)   # LOSS_REGISTRY
       │               └─ setup_optimizers() / setup_schedulers()
       └─ training loop:
            for data in train_loader:
                model.feed_data(data)
                model.optimize_parameters(current_iter)   # forward → loss → backward → step → EMA
                logger 记录 loss
                if current_iter % val_freq == 0:
                    model.validation(...)                 # nondist_validation / dist_validation
                    model.save(epoch, current_iter)       # net_g_xxx.pth + xxx.state
```

一次 `optimize_parameters`（`basicsr/models/sr_model.py:92`）的完整内容：`zero_grad → net_g(lq) → cri_pix/cri_perceptual 求和 → backward → optimizer.step() → reduce_loss_dict → model_ema`。**改训练逻辑（新 loss、多网络、特殊 feed）就是继承 `SRModel` 重写这几个方法。**

`BaseModel`（`basicsr/models/base_model.py`）已提供：`get_optimizer`/`setup_schedulers`/`update_learning_rate`、`model_ema`、`save_network`/`load_network`（支持 `param_key` 与 non-strict）、`save_training_state`/`resume_training`、`reduce_loss_dict`（多卡 loss 平均）。子类中 `pass` 的方法（`feed_data`、`optimize_parameters`、`save` 等）是必须实现的抽象钩子。

### 3.4 测试 / 推理流程（PDF §3.3–3.4）

- `basicsr/test.py`：复用同一套 build_dataset/build_model；加载 `path.pretrain_network_g`；对每个 test dataset 调 `model.validation()` 输出指标，`val.save_img: true` 保存结果图到 `experiments/<name>/visualization/<dataset>/`。
- `inference/*.py`：脱离配置的轻量推理脚本范例（加载 pth → 单图前向 → 保存），适合改造成去雾 demo 脚本。

## 4. 配置系统（YAML）

模板：`options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml`。解析器：`basicsr/utils/options.py::parse_options`。

训练配置的主要段落：

| 段落 | 作用 | 关键键 |
|---|---|---|
| 头部 | 实验名/模型/scale/GPU/seed | `name`（含 `debug` 字样即进入 debug 模式）、`model_type`、`scale`、`num_gpu` |
| `datasets.train` | 训练数据 | `type: PairedImageDataset`、`dataroot_gt`/`dataroot_lq`、`gt_size`（GT 裁剪尺寸）、`use_hflip`/`use_rot`、`batch_size_per_gpu`、`num_worker_per_gpu`、`dataset_enlarge_ratio`、`io_backend: {type: disk\|lmdb}`、`meta_info_file`（可选） |
| `datasets.val` | 验证数据（可多个，如 val_2） | 同上没有增强项 |
| `network_g` | 生成网络 | `type` 即 arch 类名，其余为构造参数 |
| `path` | 预训练/续训 | `pretrain_network_g`、`param_key_g`（`params` 或 `params_ema`）、`strict_load_g`、`resume_state` |
| `train` | 训练策略 | `ema_decay`、`optim_g`、`scheduler`、`total_iter`、`warmup_iter`、`pixel_opt`、`perceptual_opt` |
| `val` | 验证频率与指标 | `val_freq`、`save_img`、`metrics.<name>.type`（指标函数名）+ 指标参数 |
| `logger` | 日志 | `print_freq`、`save_checkpoint_freq`、`use_tb_logger`、`use_swanlab`、`swanlab` |
| `dist_params` | 分布式 | `backend: nccl`、`port` |

实验命名约定（PDF §4.3.1.1）：`<编号>_<模型>_<关键配置>_<数据集>_<迭代数>k_B<batch>G<卡数>`，如 `001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1`。

**SwanLab 实验跟踪（开箱即用）**：配置 `use_swanlab: true` 即启用，框架通过 `swanlab.sync_tensorboard_torch()` 自动同步 tensorboard 的全部标量（loss、lr、验证指标）与图像到 SwanLab，训练代码零改动。`logger.swanlab` 配置段：`project`（为空取实验名）、`mode`（默认 `local`，本地记录无需账号，用 `swanlab watch <logdir>` 打开看板；`online` 同步云端需先 `swanlab login`）、`resume_id`（续传到已有实验）、`logdir`（默认 `./swanlog`）。前置要求：`use_tb_logger: true`，且 debug 模式自动关闭。详见 `basicsr/utils/logger.py::init_swanlab_logger`。

## 5. 开发指引：如何添加/修改模块

通用四步（PDF 第 6 章，所有模块类同）：
1. **放文件**：按后缀约定放到对应目录；
2. **写实现**：类名/函数名不得与已有重名；
3. **注册**：加对应 Registry 的装饰器（metrics 例外，见下）；
4. **写配置**：在 yml 的对应 `type` 字段填类名/函数名。

### 5.1 添加网络结构（arch）

```python
# basicsr/archs/mydehaze_arch.py   （文件名以 _arch.py 结尾）
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights   # 可复用的初始化/构件工具

@ARCH_REGISTRY.register()
class MyDehazeNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48):
        ...
```

yml 中：`network_g: {type: MyDehazeNet, num_feat: 48, ...}`。
构建顺序参考：`srresnet_arch.py`（卷积骨干）、`rrdbnet_arch.py`（残差稠密块）、`swinir_arch.py`（Transformer，`arch_util.trunc_normal_` 初始化）。

### 5.2 添加模型（model）——去雾的主战场

```python
# basicsr/models/dehaze_model.py   （文件名以 _model.py 结尾）
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel

@MODEL_REGISTRY.register()
class DehazeModel(SRModel):
    # 大多数情况只需重写需要变动的钩子，例如：
    # feed_data: 改输入数据的组装（如额外读 depth/transmission）
    # optimize_parameters: 改 loss 组合/训练技巧
    # test: 改推理策略（如大图分块）
    ...
```

yml 中：`model_type: DehazeModel`。
**去雾与超分的唯一框架级差异是 `scale`**：`SRModel` 本身不写死 scale，纯 LQ→GT 映射。去雾配置写 `scale: 1` 即可（dataset 的 `paired_random_crop` 仍需要 scale 字段，取 1 表示同尺寸裁剪）。

### 5.3 添加数据集（dataset）

- 常规情况**不用写**：直接把 `dataroot_gt`/`dataroot_lq` 指到 `datasets/` 下的雾图/清晰图文件夹即可（子图切分用 `scripts/data_preparation/extract_subimages.py`，大图加速用 `create_lmdb.py` 生成 lmdb 后 `io_backend: {type: lmdb}`）。
- 需要自定义（如同步雾合成、读附加模态）：

```python
# basicsr/data/my_dataset.py   （文件名以 _dataset.py 结尾）
@DATASET_REGISTRY.register()
class MyDataset(data.Dataset):
    def __getitem__(self, index):
        ...
        return {'lq': img_lq, 'gt': img_gt}   # 约定：lq/gt 为 CHW float32 tensor [0,1]，BGR→RGB 已转换
```

返回 dict 的键会被 `model.feed_data(data)` 消费；`SRModel.feed_data` 读取 `data['lq']` 和可选的 `data['gt']`。
图像处理工具：`basicsr/utils/img_util.py`（`imfrombytes`/`img2tensor`/`tensor2img`/`imwrite`）、`basicsr/data/transforms.py`（`augment`、`paired_random_crop`）。

### 5.4 添加损失函数

```python
# basicsr/losses/my_loss.py   （文件名以 _loss.py 结尾）
@LOSS_REGISTRY.register()
class MyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ...):
        ...
```

yml 中：`train: {pixel_opt: {type: MyLoss, loss_weight: 1.0}, ...}`。非类形式的损失（普通函数）不走注册表，在 model 里直接 import 调用。

### 5.5 添加指标

```python
# basicsr/metrics/my_metric.py
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_my_metric(img, img2, **kwargs):
    ...
```

**必须**同步在 `basicsr/metrics/__init__.py` 中显式 import（`from .my_metric import calculate_my_metric`），否则注册不生效。yml：`val.metrics.xxx.type: calculate_my_metric`。指标一般在 numpy 上计算；torch 版本可加 `_pt` 后缀命名。

## 6. 去雾二次开发的标准路径

1. **网络**：在 `basicsr/archs/` 新建 `<name>_arch.py`，写 `nn.Module` + `@ARCH_REGISTRY.register()`。可直接 copy `rrdbnet_arch.py`/`swinir_arch.py` 改 block。
2. **模型**：若标准 L1/感知损失 + PSNR/SSIM 验证够用（大多数去雾工作如此），**直接复用 `SRModel` 不写新 model**；需要定制 loss 组合或推理策略时按 §5.2 继承重写。
3. **配置**：copy `options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml` 改名，改 `name` / `scale: 1` / `network_g` / 数据路径 / `gt_size`（= 训练 patch 尺寸）/ `total_iter`。测试 copy `options/test/` 下对应模板。
4. **数据**：按 `datasets/README.md` 组织软链接；雾图与 GT 同名成对存放。
5. **验证**：先 `--debug` 跑通，再短迭代正式跑，最后全量训练。

## 7. Git 提交约定（强制）

**每次完成修改后，必须立即用中文提交本次改动**，让所有变更可追溯、可回滚。具体要求：

- 提交前须按 §8 的配方完成验证（至少通过导入检查；涉及模型/训练/数据改动的，须通过冒烟训练）；
- 只提交与本次改动相关的文件，不夹带无关修改；工作区若存在用户进行中的其他改动，不要代为提交；
- 提交信息使用中文，格式为 `<类型>: <一句话描述>`。类型参考：`feat` 新功能 / `fix` 修复 / `refactor` 重构 / `docs` 文档 / `data` 数据与配置 / `chore` 杂项。示例：
  - `feat: 添加 FFA-Net 去雾网络结构`
  - `fix: 修复 PairedImageDataset 在 lmdb 后端下的键读取错误`
  - `docs: 更新 AGENTS.md 中 registry 机制说明`
- 描述里写清"做了什么、为什么"，不要写 `update`、`修改` 这类无信息量的内容；
- 只执行 commit，**不 push**；push、amend、rebase 等共享/改写历史的操作，必须得到用户明确指示后进行；
- 一次改动涉及多个模块时，按模块拆成多个提交，便于回溯与 cherry-pick。

## 8. 验证与冒烟测试配方（改动后必做）

`basicsr` 是 editable 安装，任何改动都直接影响运行时。改动后按顺序检查：

```bash
conda activate basicsr
python -c "import basicsr; print(basicsr.__version__)"   # 快速导入检查（新文件后缀错误、注册重名会在此暴露）
```

冒烟训练（合成数据 6 iter 全链路验证：dataset→model→loss→save→validation）：

```bash
python - <<'EOF'   # 生成合成数据
import numpy as np, cv2, os
rng = np.random.RandomState(0)
for split, n in [('train', 6), ('val', 2)]:
    os.makedirs(f'/tmp/smoke/{split}/GT', exist_ok=True)
    os.makedirs(f'/tmp/smoke/{split}/LQ', exist_ok=True)
    for i in range(n):
        gt = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        cv2.imwrite(f'/tmp/smoke/{split}/GT/{i:03d}.png', gt)
        cv2.imwrite(f'/tmp/smoke/{split}/LQ/{i:03d}.png', cv2.resize(gt, (16, 16), interpolation=cv2.INTER_CUBIC))
EOF
# 写一份配置：scale: 4, gt_size: 64, total_iter: 6, val_freq: 3, num_worker_per_gpu: 0, use_tb_logger: false
python basicsr/train.py -opt /tmp/smoke.yml            # 训练应正常迭代、保存、验证
python basicsr/test.py  -opt /tmp/smoke_test.yml       # 测试应正常加载 checkpoint、出指标
```

2026-10-12 裁剪完成后已用此配方验证通过（训练 6 iter + PSNR 验证 + checkpoint 保存/加载 + 结果图保存）。

## 9. 常见坑与约定速查

- **文件后缀即注册**：新模块文件必须以 `_arch.py`/`_model.py`/`_dataset.py`/`_loss.py` 结尾；普通工具文件避开这些后缀。
- **metrics 双重步骤**：注册装饰器 + `metrics/__init__.py` 显式 import，缺一不可。
- **无参考指标**：`calculate_fade/niqe/brisque`（越低越好，依赖 pyiqa 与 fade-python，见 requirements 注释的免依赖装法）适合真实雾图等无 GT 场景；速度较慢，建议只在最终评估开启，勿挂到每 1k iter 的训练验证上。
- **类/函数名全局唯一**：跨模块也不能重名（注册表启动时即报错）。
- **`scale` 字段**：dataset 的成对裁剪依赖它；去雾写 1。
- **EMA**：`train.ema_decay > 0` 即开启；保存的 pth 含 `params` 与 `params_ema` 两个字段，测试/推理时用 `path.param_key_g: params_ema` 加载 EMA 权重。
- **resume 两种方式**：yml 里设 `path.resume_state: <xxx.state>`（覆盖 pretrain 设置），或命令行 `--auto_resume`。
- **数据通道约定**：dataset 返回的 tensor 为 RGB、CHW、float32、[0,1]；OpenCV 读图是 BGR，用 `img2tensor(..., bgr2rgb=True)` 转换。
- **可视化**：`model.get_current_visuals()` 返回 lq/output/gt 的 GPU tensor，validation 时由框架统一 `tensor2img` 保存。
- **分布式**：多卡只需 `bash scripts/dist_train.sh N config.yml`；模型内部用 `@master_only` 控制只在主卡保存/打印，metric 求均值走 `dist_validation`。
- **SwanLab**：`use_swanlab: true` 同步 tensorboard 指标到 SwanLab（依赖 `swanlab[dashboard]`）；本地看板用 `swanlab watch swanlog` 查看；云端模式先 `swanlab login` 再设 `mode: online`。
- **被删内容的线索**：若某篇论文代码引用了 `basicsr.ops`、`video_*`、`stylegan2_*`、`realesrgan*` 等，说明它依赖已裁剪模块，需要先从 git 历史恢复（`git checkout <commit> -- <path>`）或改用保留的等价物。
- **SD-LoRA 去雾（SDLoRADehaze/SDLoRAModel）**：SD 权重从 `sd_path`（默认 `experiments/pretrained_models/sd21base`，下载见 `research/002_sdlora/download_sd21base.sh`）加载；checkpoint 只含 LoRA 参数（MB 级），加载/续训必须 `strict_load_g: false`；输入需 pad 到 8 的倍数（SDLoRAModel.test 已处理）；`fixed_timestep` 勿用 999（ᾱ≈0.005 导致 x0 重建放大 ~14.6 倍、纯像素监督不收敛，实测见 research/002_sdlora/RESEARCH_LOG.md），用 250 或 `prediction: x0`；OWH 各子集图片尺寸不齐，训练集需先用 `scripts/data_preparation/filter_min_size.py` 按 gt_size 过滤。
- **临时素材勿放 /tmp**：`/tmp` 会被系统定期清理，冒烟数据集等需持久的小样本放 `research/` 下。

## 10. 与 BasicSR-Tutorial-v0.5.pdf 章节对照

| 本文件 | PDF 章节 | 内容 |
|---|---|---|
| §3.2 | §4.2 | Registry 注册机制、自动扫描、后缀约定、重名问题 |
| §3.3 | §3.2 | 训练流程（入口、dataset/model 创建、训练循环） |
| §3.3 | §3.3–3.4 | 测试流程、推理流程 |
| §3.4 / §4 | §4.3 / §4.5 | 配置系统 / BaseModel 与 SRModel 方法详解 |
| §5 | 第 6 章 | 如何添加与修改 Dataset/模型/网络/损失/指标 |
| §6 | §3.1、第 7 章 | 目录解读、数据准备（lmdb、meta、FileClient） |
| §9 | §4.5.4–4.5.6 | 保存/resume、validation、EMA |
| — | 第 5 章 | PSNR/SSIM 指标原理（仍可参考，NIQE 已删除） |
| — | §4.8、第 8/9/10 章 | CUDA 算子、部署、脚本、examples 模板（对应模块已裁剪/删除，仅作背景了解） |
