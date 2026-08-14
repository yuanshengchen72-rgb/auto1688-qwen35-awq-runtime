# Auto1688 Qwen3.5 AWQ Runtime

这是 1688 搜同款桌面软件在 AutoDL Art 上使用的最小公开运行时仓库。仓库不包含商品业务源码、桌面软件源码、账号凭据、用户数据或模型权重，只负责验证镜像、恢复固定 vLLM 环境，以及把 AutoDL Art 公共模型链接到固定运行目录。

固定运行契约：

- 模型仓库：`cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit`
- 服务模型名：`Qwen3.5-35B-A3B-AWQ-4bit`
- 量化：AWQ 4-bit / `compressed-tensors`
- vLLM：`0.23.1rc1.dev1061+g36484e464`
- Python：3.12
- 模型文件：20 个，其中 5 个 `safetensors` 分片

AutoDL Art 镜像审核可执行：

```bash
cd /root/auto1688-qwen35-awq-runtime
PYTHONPATH=src python -m auto1688_art_runtime check-image
```

桌面软件部署时可执行完整准备：

```bash
cd /root/auto1688-qwen35-awq-runtime
PYTHONPATH=src /root/autodl-tmp/qwen35-env/bin/python -m auto1688_art_runtime prepare
```

`prepare` 的外部接口不接受模型名或版本参数，避免调用方误选模型。它会验证镜像内 `/root/qwen35-env.tar.zst`、恢复固定 Python/vLLM 环境、读取 `/root/auto1688-art-model-manifest.json`，并原子建立 `/root/autodl-tmp/models/Qwen3.5-35B-A3B-AWQ-4bit`。

模型文件遵循其上游仓库内的 Apache-2.0 许可证；本仓库中的运行时脚本使用 MIT 许可证。发布模型时必须一并保留上游 `LICENSE`、版权和必要声明。
