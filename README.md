# Fiscal — 《伪人降临》伏笔考据与 GPT-SoVITS「曼波」配音合成

本仓库收录《伪人降临！我是全场唯一预言家》核心伏笔考据文档、抖音「曼波」解说配音台本，以及基于 `RVC-Boss/GPT-SoVITS` 的自动化语音合成脚本与成品音频。

## 目录结构

- `weiren_foreshadowing_summary.md`：《伪人降临！我是全场唯一预言家》（第 1–442 章）九大核心伏笔与原文考据汇总
- `kl_origin_mambo_script.md`：《全书没一个活人？揭秘神秘组织 KL 的催泪真相》抖音「曼波」分段配音台本
- `synthesize_kl_mambo.py`：基于 `GPT-SoVITS` 的 GPU 批量分段合成、静音裁剪与响度归一化脚本
- `remote_setup_and_run.sh`：远端 GPU 节点环境初始化、模型下载与推理一键执行脚本
- `kl_origin_mambo_douyin.wav` / `.mp3`：抖音紧凑解说版成品音频（1.08x 语速）
- `kl_origin_mambo_standard.wav` / `.mp3`：标准语速版成品音频（1.00x 语速）
