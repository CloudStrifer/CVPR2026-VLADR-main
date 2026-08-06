# Cross-Category Continual CLIP-ReID Baseline

This working copy is derived from the CVPR 2026 VLADR repository:
*Vision-Language Attribute Disentanglement and Reinforcement for Lifelong
Person Re-Identification*.

The active implementation has been reduced to the global components required
for cross-category continual ReID. The pre-cleanup documentation and source
files are preserved with the `_delete` suffix.

## Active method

The code now contains:

- CLIP ViT-B/16 image encoder;
- frozen CLIP text encoder;
- one learnable global prompt for every identity;
- per-domain object nouns in the identity Prompt;
- optional fixed visual-prototype, category-centered, and full OCIA identity
  anchors;
- identity cross-entropy loss;
- Triplet loss on the 512-D CLIP image projection;
- global image-prompt alignment loss `L_global`;
- optional Semantic Compatibility-Aware Selective Distillation (`L_SCSD`);
- one private visual Adapter for every training domain, including domain 1;
- optional OCIA-guided semantic-debiased Adapter fusion (OSAF) for unseen
  domains;
- sequential domain training, classifier expansion, prompt-bank expansion,
  checkpointing, and evaluation on seen and held-out unseen domains.

The canonical Chinese method, training, ablation, parameter-sweep, and
unseen-domain evaluation guide is
[docs/ocia_osaf_complete_guide_zh.md](docs/ocia_osaf_complete_guide_zh.md).
Use it as the primary entry point; the other files in `docs/` preserve the
method's design history or explain individual optional modules.

The Stage 2 objective is:

```text
L_total = L_CE + L_Triplet + lambda_global * L_global
          + lambda_scsd * L_SCSD
          + lambda_relation * L_global_relation
```

The optional anchor modes only replace the fixed target of `L_global`. See
[docs/object_aware_cross_modal_identity_anchoring_zh.md](docs/object_aware_cross_modal_identity_anchoring_zh.md)
for the complete OCIA method, switches, checkpoint requirements, and ablation
commands. The earlier category-centering baseline remains documented in
[docs/category_prompt_and_visual_residual_zh.md](docs/category_prompt_and_visual_residual_zh.md).

SCSD uses frozen-CLIP attribute queries, a previous-stage frozen teacher, and
semantic routing to transfer only compatible cross-category relations. See
[docs/semantic_compatibility_aware_selective_distillation_zh.md](docs/semantic_compatibility_aware_selective_distillation_zh.md).

BLIP descriptions, instance-level `L_MAlign`, the strong-augmentation duplicate
branch, and post-domain model fusion are not part of the active path. The
original index-locked `L_DAlign` is retained only as the `index` ablation;
`scsd` is the cross-category distillation path.

The domain-adapter mode isolates old visual paths after the first domain and
provides the active anti-forgetting path. OSAF reuses only category-debiased
Adapter residuals when evaluating a held-out unseen domain.

## Installation

```shell
conda create -n continual-clipreid python=3.9
conda activate continual-clipreid
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python setup.py develop
```

## Cross-category data format

Use one CSV, TSV, or JSONL manifest per domain. Every row needs:

```text
path,pid,camid,split
images/train/0001_a.jpg,identity_0001,0,train
images/query/0101_a.jpg,identity_0101,0,query
images/gallery/0101_b.jpg,identity_0101,1,gallery
```

`split` must be `train`, `query`, or `gallery`. The five-domain configuration
template is [config/cross_category_example.json](config/cross_category_example.json).

Validate the manifests before GPU training:

```shell
python tools/validate_cross_category_config.py \
  --domain-config config/cross_category_example.json \
  --data-dir /path/to/cross_category_reid
```

## Stage 1: global identity prompts

Train a separate identity-prompt checkpoint for every training domain:

```shell
python train_stage1.py \
  --domain-config config/cross_category_example.json \
  --data-dir /path/to/cross_category_reid \
  --stage1-prompts-out-dir ./_CROSS_CATEGORY_PROMPTS
```

The `prompt_checkpoint` paths in the domain configuration must point to the
same output files. Current Stage 1 checkpoints also store frozen-CLIP visual
identity prototypes, the domain category center, and the fixed OCIA category
semantic subspace.

## Stage 2: sequential training

```shell
python train_stage2.py \
  --domain-config config/cross_category_example.json \
  --data-dir /path/to/cross_category_reid \
  --stage1-prompts-out-dir ./_CROSS_CATEGORY_PROMPTS \
  --visual-anchor-mode ocia \
  --anchor-residual-weight 1.0 \
  --anchor-temperature 0.07 \
  --classifier-lr-multiplier 10.0 \
  --attr-distill-mode none \
  --classifier-scope current \
  --eval-descriptor raw \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-lr 3e-4 \
  --adapter-routing osaf \
  --adapter-routing-scope all \
  --adapter-topk 2 \
  --adapter-semantic-weight 0.5 \
  --adapter-routing-temperature 0.1 \
  --adapter-debias-strength 1.0 \
  --adapter-fusion-weight 1.0 \
  --logs-dir ./_RESULTS \
  --eval-stage 0,1,2,3,4
```

The default evaluation descriptor concatenates the raw 768-D global CLIP
feature and the raw 512-D CLIP projection, avoiding cross-category BN running-
statistics drift. The legacy BN descriptor remains available by switch.

To freeze the shared visual backbone after the first domain and allocate one
private visual Adapter for every domain, including the first domain, use:

```shell
--continual-update-mode domain_adapter \
--adapter-last-blocks 6 \
--adapter-bottleneck-dim 128 \
--adapter-lr 3e-4 \
--adapter-routing osaf \
--adapter-routing-scope all
```

By default, seen domains use their corresponding full Adapter and held-out
unseen domains use OSAF. Add `--adapter-routing-scope all` to route both seen
and unseen evaluation images through OSAF Top-K fusion. This reports
domain-agnostic `Seen-Routed` performance without changing training. See
[docs/ocia_osaf_complete_guide_zh.md](docs/ocia_osaf_complete_guide_zh.md)
for the complete command, behavior, and ablations.

Legacy person-ReID orders remain available through `--setting`. Example shell
scripts are provided in `train1.sh`, `train2_setting1.sh`,
`train2_setting2.sh`, and `test.sh`.

## Backup convention

- Partially simplified files: the pre-cleanup version is stored beside the
  active file as `name_delete.py`, while `name.py` contains the global
  baseline.
- Fully disabled modules: only the renamed `name_delete.py` remains.
- Existing BLIP text and prompt output directories were not deleted.

## Original paper

- [arXiv:2603.19678](https://arxiv.org/abs/2603.19678)

```bibtex
@inproceedings{xu2026vladr,
  title={Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification},
  author={Xu, Kunlun and Cheng, Haotong and Li, Jiangmeng and Zou, Xu and Zhou, Jiahuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
