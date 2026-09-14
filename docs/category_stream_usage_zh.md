# 第 1 步：类别交错数据流使用说明

**2026-09-14 更新：** 已另外生成真实五数据集、四阶段配置 `config/category_progressive_real/main.json` 和全部 CSV，参见 [上传与使用说明](real_data_stream_upload_zh.md)。下文保留第 1 步时的协议说明与占位示例；不再需要从零手写真实清单。

本功能实现数据协议、审计和加载器，尚未接入 ECPM、PGCA 或新训练入口。

实现文件：

- [协议与审计](E:/Multi_modal_Code/CVPR2026-VLADR-main/lreid_dataset/category_stream.py)
- [当前阶段加载器](E:/Multi_modal_Code/CVPR2026-VLADR-main/lreid_dataset/category_stream_loaders.py)
- [审计命令](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/validate_category_stream.py)
- [三阶段示例配置](E:/Multi_modal_Code/CVPR2026-VLADR-main/config/category_progressive_example.json)
- [中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step1_progress.md)

## 1. 先验证示例

在仓库根目录执行：

```powershell
python tools/validate_category_stream.py --stream-config config/category_progressive_example.json --output docs/category_stream_example_audit.json
```

示例为：

| 阶段 | 当前数据 | 新类别 | 重复类别 | 历史中暂时缺席类别 |
|---|---|---|---|---|
| t1 | person a/b；vehicle a/b | person、vehicle | 无 | 无 |
| t2 | person c/d；panda a/b | panda | person | vehicle |
| t3 | vehicle c/d；person e/f | 无 | person、vehicle | panda |

共 12 个训练身份，每个身份 2 张图像条目。另有独立的验证/测试条目。不同类别的 a/b 不是同一身份。

**示例仅包含元数据，图片路径是占位路径。** 默认审计只读配置和 manifest，不要求占位图片存在，不运行模型，不证明真实数据已经准备好。未配置某类别的验证集或测试集会给出 NOTE，不会虚构评估结果。

准备真实数据后，再加文件存在性检查：

```powershell
python tools/validate_category_stream.py --stream-config config/my_category_stream.json --check-images --output docs/my_category_stream_audit.json
```

`my_category_stream.json` 是你需要准备的真实配置名称，不是已经生成的文件。校验失败退出码为 2；检查通过退出码为 0。`--check-images` 只检查文件存在，不解码图片。只有迭代图像 loader 时才进行解码。

## 2. JSON 配置规则

支持 `schema_version: 1` 的 JSON 配置。未知配置字段会报错，避免字段拼错后静默失效。

顶层字段：

| 字段 | 含义 |
|---|---|
| schema_version | 必须是整数 1 |
| description | 可选说明，不影响学习 |
| data_root | 图像默认根目录；每个训练/评估条目可覆盖 |
| stages | 非空阶段列表，列表顺序就是时间顺序 |
| evaluation | 可选的固定评估集合，区分 validation/test |
| identity_aliases | 可选的跨数据源真实身份映射 |

每个阶段包含 `stage_id` 和非空 `categories`；阶段编号唯一，每个阶段内一个类别只能有一个条目。每个类别条目包含 `category`、`train_manifest`，可选 `data_root`。

路径统一规则：

- `train_manifest`、`query_manifest`、`gallery_manifest` 相对 **配置所在目录**。
- `data_root` 也相对 **配置所在目录**，不是相对 manifest。
- manifest 的 `path` 相对该条目最终使用的 `data_root`。
- 上述字段均可使用绝对路径。运行命令时切换工作目录不会改变解析结果。

`data_root` 必须在顶层或相应条目中显式指定。旧 domain 配置中的 `root` 不会自动识别为 `data_root`。

## 3. Manifest 与身份规则

支持 CSV、TSV、JSONL。每行最少包含：

```csv
path,original_pid,source_dataset,camid
person/a_1.jpg,a,person_demo,0
person/a_2.jpg,a,person_demo,1
person/b_1.jpg,b,person_demo,0
person/b_2.jpg,b,person_demo,1
```

- `original_pid` 保留原始身份，而不是每阶段重新编号后的标签；字符串 `001` 和 `1` 不会被强行合并。
- `source_dataset` 是稳定的数据来源标识，同一数据集跨阶段保持不变，不能写成 `person_t1`、`person_t2` 来绕过重复检查。
- `camid` 是实际相机/协议分组编号，保留为字符串；是否排除同相机由评估协议决定。
- 可选的 `category`、`stage_id`、`split` 若填写，必须与父配置一致。
- 每个训练 manifest 只放该阶段、该类别的训练行。不能将旧 manifest 的 train/query/gallery 混在一起直接传入。
- 空 manifest 报错；同一 manifest 内重复图片路径报错。

默认身份键为：

```text
(category, source_dataset, original_pid)
```

`stage_id` 不在身份键里。这让审计能够发现同一身份跨阶段重复。

同一类别在各阶段分别建立局部标签 `0...N-1`。局部标签只供临时身份分类器使用；ECPM 和跨阶段身份审计使用完整 `identity_key`。

### 将已有数据准备成新协议

仓库现有 manifest 使用 `pid`、`domain` 等字段，不能未经转换直接作为本协议输入。推荐离线准备顺序：

1. 先保留现有 train/query/gallery 划分，不把已有测试身份挪回训练。
2. 明确哪些训练身份留作验证集；如新增验证划分，先按完整身份划分，再生成 query/gallery。
3. 对剩余训练身份按 `(category, source_dataset, original_pid)` 分组。
4. 固定随机种子或显式分配表，将整个身份组分配到一个阶段；该身份所有训练图像一起移动。
5. 将旧 `pid` 写入 `original_pid`，旧真实数据来源写入 `source_dataset`；不要使用旧 loader 重映射后的临时 PID。
6. 输出每阶段每类别 manifest，query/gallery 单独输出，运行审计。

本次未自动重划你现有的五数据集文件，也没有决定正式实验的阶段比例、随机种子和验证身份。示例用于检查数据结构；正式划分应另存配置和 manifest，保持原始划分可追溯。

### 跨数据源的同一个真实身份

若两个来源确实包含同一个人，显式映射到相同 `canonical_id`：

```json
"identity_aliases": [
  {"category": "person", "source_dataset": "dataset_a", "original_pid": "12", "canonical_id": "person_0007"},
  {"category": "person", "source_dataset": "dataset_b", "original_pid": "93", "canonical_id": "person_0007"}
]
```

映射后的身份键为 `(category, "__canonical__", canonical_id)`；两个来源的原始字段仍保留。映射后若该真实身份跨阶段或跨 train/validation/test 出现，同样报错。写错且未匹配到任何行的 alias 也会报错。

审计能检查身份元数据和解析后图片路径，不能仅凭元数据发现被复制改名的相同图片、漏标的真实身份关系。数据整理时仍需正确提供身份信息。

## 4. 固定验证与测试集合

每个评估条目示例：

```json
{
  "name": "person_test",
  "category": "person",
  "split": "test",
  "protocol": "cross_camera",
  "query_manifest": "manifests/person_test_query.csv",
  "gallery_manifest": "manifests/person_test_gallery.csv"
}
```

规则：

- `split` 为 `validation` 或 `test`；各集合的真实身份与全部阶段训练身份互斥，验证身份和测试身份也互斥。
- 每个评估 `name` 唯一。本版本不同命名评估集合之间也要求身份互斥，避免重复计数；需要共享同一身份集的额外诊断时，复用同一个 EvaluationView。
- query/gallery 共享一套局部 PID 映射。
- 同一固定评估集合可跨训练阶段反复评估。
- `stream.evaluations_at(stage_id, split)` 只暴露截至该阶段已到达类别的对应评估集合。
- 当前协议要求评估类别最终出现在训练流中；完全未见物体类别的扩展评估留到后续步骤。

两种显式检索规则：

| protocol | 有效正样本条件 |
|---|---|
| cross_camera | 同一 identity_key、不同图片、不同 camid |
| exclude_self | 同一 identity_key、不同图片；允许同 camid |

每个 query 至少有一个有效 gallery 正样本。query 图片可以同时出现在 gallery，但它自己不能成为唯一有效正样本。

只有当数据集允许同相机匹配时才用 `exclude_self`，不能为了让审计通过而改变协议。若跨来源 alias 合并同一真实身份，需要先统一其真实相机编号；来源名称变化本身不能算作换相机。复杂 session/cross-view 协议应先核实其过滤规则能否表达为上述规则，不自动将旧 `eval_protocol` 名称猜测映射过来。

## 5. 后续代码如何读取当前阶段

下面是接口示例；替换为真实配置后才可迭代图片。Windows 使用多进程 loader 时保留 `if __name__ == '__main__'`。

```python
import random
import numpy as np
import torch

from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    stream = load_category_stream('config/my_category_stream.json')
    stage = stream.stage('t2')
    loaders = build_stage_loaders(
        stage,
        batch_size=4,
        num_instances=2,
        prototype_batch_size=128,
        workers=0,
        seed=seed,
        transform_options={'height': 224, 'width': 224, 'resize_mode': 'pad'},
    )

    for category, pair in loaders.items():
        print(category, pair.view.label_map)
        pair.set_epoch(0)
        train_batch = next(iter(pair.train))
        # 第 3 步使用 images/targets 学习当前类别。
        print(train_batch['images'].shape, train_batch['targets'])

        for batch in pair.prototype:
            # 第 4 步使用 images 提取参考特征，按 identity_keys 聚合。
            print(batch['identity_keys'])

    # 阶段结束后释放这些 loader，不传入下一阶段。
    del loaders


if __name__ == '__main__':
    main()
```

batch 字段：

| 字段 | 类型/含义 |
|---|---|
| images | Tensor [B,C,H,W] |
| targets | LongTensor [B]，当前 stage/category 的局部身份标签 |
| identity_keys | 长度 B 的完整身份键列表，不按列转置 |
| paths | 长度 B 的绝对图片路径列表 |
| categories / stage_ids | 物体类别和所属训练阶段 |
| source_datasets / original_pids / camids | 原始元数据 |
| splits | 训练为 train，评估为 query/gallery |

训练 loader 使用每类别 P×K batch，要求 P≥2、K≥2，类别身份不足时报错；不从其他类别找负样本。其采样行为与旧 P×K 规则一致，另用了局部 RNG 和显式 epoch，避免修改全局采样随机状态。每个 epoch 调用 `pair.set_epoch(epoch)`。

原型 loader 顺序遍历当前类别全部训练行，`drop_last=False`，每张图一次，不使用训练 sampler。默认参考变换只有 resize、ToTensor 和 Normalize。默认归一化沿用仓库设置；第 2 步选定参考编码器后应固定并记录同一套参数。

`seed` 控制 P×K 采样及 worker 随机种子；`workers=0` 的随机图像增强使用主进程 RNG，因此训练入口仍需像示例一样设定 Python/NumPy/PyTorch 随机种子。完整训练断点和 RNG 保存将在第 8 步实现，本步骤不声称实现训练断点恢复。

`build_evaluation_loaders(evaluation, reference_transform, ...)` 只构建评估 loader。评估 batch 保留类别用于算指标，但第 9 步的路由和模型只接收 `batch['images']`，不得读取真实类别。

## 6. 可重复验证与恢复

```powershell
python -m unittest discover -s tests -p "test_category_stream*.py" -v
```

测试使用临时图片和真实 DataLoader，覆盖身份泄漏、别名、路径重复、评估过滤、P×K、原型全覆盖、当前阶段隔离与 Windows 子进程。无需 GPU 或真实训练数据。

若额度中断，后续让 Codex 先读取 [执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step1_progress.md)，核实记录中的已完成段和实际文件，再继续未完成项。代码、示例、测试和审计报告都已落盘，不依赖上一轮进程仍在运行。
