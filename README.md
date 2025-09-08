# FlashAttention (Metal Port)

This repository ports the official implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention) to Apple silicon. It is a minimal, maintainable set of source files that reproduces the FlashAttention algorithm.

## Python/MLX Port

This repository now includes a Python port of the FlashAttention algorithm using Apple's MLX framework. This implementation provides a pure Python alternative to the original Swift/Metal version, allowing for easier integration with Python-based machine learning projects on Apple Silicon.

### Usage

#### Environment Setup

It is recommended to use a virtual environment.

<details>
<summary>Using `uv` (Recommended)</summary>

This project uses `uv` for environment and package management.

1. **Create a virtual environment:**

    ```bash
    uv venv
    ```

2. **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    uv pip install -e .
    ```

</details>

<details>
<summary>Using `pip` and `venv`</summary>

1. **Create a virtual environment:**

    ```bash
    python -m venv .venv
    ```

2. **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -e .
    ```

</details>

#### Running Tests

To verify the implementation, you can run the provided tests:

```bash
python -m tests.test_attention
```

The tests compare the output of the custom FlashAttention implementation against a standard attention implementation to ensure numerical correctness.

---
