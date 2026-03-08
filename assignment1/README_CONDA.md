# Conda 使用说明

## 环境设置

```bash
cd assignment1
bash scripts/setup_conda.sh
```

这会创建名为 `ntt_jax` 的 conda 环境并安装所有依赖。

## 激活环境

```bash
conda activate ntt_jax
```

## 运行测试

```bash
pytest
pytest --logn 10 --batch 4
```

## 运行基准测试

```bash
python -m tests.benchmark
python -m tests.benchmark --tests --logn 10 --batch 4
python -m tests.benchmark --bench --logn 12 --batch 4
```

## 生成提交文件

```bash
bash scripts/make_submission_conda.sh
```
