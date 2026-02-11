### 🌟 简介
---
本项目是对论文 [DOI: 10.3847/1538-4357/abb5b6](https://www.doi.org/10.3847/1538-4357/abb5b6) 的代码复现。

代码实现参考自：[Zenodo](https://zenodo.org/records/3979217)。


### 🛠️ 推荐环境配置
---
为确保模型能够正常训练与推理，建议使用以下环境：

* **Python**: `3.10.*`
* **TensorFlow**: `2.13.0`
* **CUDA**: `11.8`
* **cuDNN**: `8.6.0`


### ❓ 常见问题与解决方案
---
在运行过程中，若遇到 `ValueError: None values not supported.` 报错，请按照以下步骤修改代码：

**Step 1**: 找到原始的优化器定义行并将其注释：
```python
# opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
```

**Step 2**: 在其下方添加优化后的定义：
```python
opt = keras.optimizers.Adam(learning_rate=1e-4)
```


### 📜 开源协议
---
本项目遵循 **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** 协议。
