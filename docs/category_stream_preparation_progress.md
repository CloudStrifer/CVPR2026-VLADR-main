# 真实数据流配置准备记录

日期：2026-09-14。已完成。

- [x] 发现仓库已有五数据集的真实 train/query/gallery manifest 和本地图片。
- [x] 确定默认方案：原始四阶段安排、seed=42、约 10% 原训练身份留作 validation，原 test 划分保持。
- [x] 实现按完整身份划分的确定性转换器，保留原 PID 和摄像头/会话标识。
- [x] 生成实际 main.json、阶段 CSV、validation/test CSV、完整身份分配及审计报告。
- [x] 校验全部图片路径、无身份泄漏、原 test 行保持、训练 P≥8；验证可迁移相对路径。
- [x] 上传包、说明及操作手册中的可用配置入口。

输入：`config/cross_category_five_domains.json` 与 `config/manifests/{market1501,veri,ipanda50,atrw,boat}.csv`。

数据实际存在。Market1501 / VeRi / iPanda50 / boat 用跨摄像头、会话或视角协议；ATRW 训练相机未知，validation 使用 exclude_self，不伪造摄像头。boat 验证只使用 augmentation=original 的图像，被选验证身份的其余增强图像也从训练中移除。

原始清单不改动。新文件写到 `config/category_progressive_real/`，图像路径相对仓库 `data/`，便于整个目录迁移到 Ubuntu。该分配是可执行的首版实验划分，不声称已确定最优阶段比例。

## 完成结果与中断接续

用户明确回复“按这个方案生成”。生成及全量审计已在额度中断前完成；恢复后核对现有文件，没有重新随机划分，继续完成说明和上传包。

- 训练：1,392 个身份、55,853 张图片条目；验证：157 个身份。
- T1 person 225 / vehicle 173；T2 person 225 / panda 16 / tiger 48；T3 vehicle 173 / panda 15 / person 225；T4 tiger 48 / boat 72 / vehicle 172。
- 共 31 份 CSV：11 份阶段训练清单，20 份 validation/test query/gallery 清单。
- 原训练身份共 1,549 个，全部且仅分配一次到训练或验证。boat 验证身份的 318 张增强图像退出训练且不进入验证。
- `docs/category_stream_builder_tests.txt`：6 项确定性、整身份、测试保留、路径迁移、增强过滤、未知相机和失败不发布检查通过。
- `config/category_progressive_real/audit_report.json`：全量图片存在性检查通过，warnings 为空；没有解码全部图像内容。
- `docs/category_real_conversion_verification.txt`：10 份原测试 query/gallery 转换后逐行一致，所有原训练身份分配唯一。
- `docs/category_progressive_real_upload.zip`：配置补充包，含相对目录结构、配方、转换器和说明；不含图像、权重或完整训练代码。
- `docs/real_data_stream_upload_zh.md`：可直接使用的配置位置、服务器目录、审计、短训练与真实数据消融命令。

本任务无待完成的数据配置工作。后续若服务器实际目录不同，按用户提供的目录调整 `data_root` 或各类别根目录，并在 Ubuntu 上重新做文件存在性审计和短训练；不要将本机审计指纹误认为跨环境续训凭证。未启动完整真实数据 GPU 实验。
