# Qwen3 法律助手项目完整说明文档

## 项目概述

本项目是基于Qwen3大语言模型的法律领域微调项目，实现了法律智能助手功能，能够提供专业的法律咨询、案件分析、合同审查等多种法律场景服务。

### 核心特性

- 🏥 **多场景医疗咨询**：支持10种法律场景的专业咨询
- 🔧 **支持LoRA微调**：支持高效的LoRA微调
- 📊 **完整训练流程**：从数据准备到模型部署的完整pipeline
- 🎯 **专业医疗提示词**：针对不同法律场景优化的提示词模板

## 项目结构

```
week9/项目实战/
├── 📁 核心脚本
│   ├── legal_assistant.py        # 法律助手主程序
│   ├── train_lora.py             # LoRA微调训练脚本
│   ├── inference.py              # 基础推理脚本
│   ├── inference_lora.py         # LoRA推理脚本
│   └── download_model.py         # 模型下载脚本
│   └── data_transfer.py          # 数据处理脚本
```

## 技术架构

### 基础模型
- **Qwen3-1.7B**: 项目主要使用的轻量级模型

### 微调技术
- **LoRA微调**: 低秩适应，高效微调技术

### 推理风格
- **法律专业提示词**: 针对不同法律场景优化

## 环境要求

### 硬件要求（基于Qwen3-1.7B）
- **LoRA微调**: 32GB显存

### 软件依赖
```bash
swanlab                    # 训练监控
modelscope==1.22.0        # 模型下载
transformers              # 模型加载
datasets==3.2.0           # 数据处理
peft                      # LoRA微调
accelerate                # 训练加速
pandas                    # 数据处理
addict                    # 配置管理
```

## 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python data_transfer.py
```
自动完成：
- 数据集处理(处理好的法律问答数据集CrimeKgAssitant.json)
- 数据预处理和格式化
- 训练/验证集划分（9:1比例）

### 3. 模型训练

#### LoRA微调
```bash
python train_lora.py
```

### 4. 模型推理

#### 基础推理
```bash
python inference.py
```

#### 医疗助手交互
```bash
python legal_assistant.py
```

## 法律场景支持

项目支持10种专业法律场景：

| 场景ID | 场景名称 | 专业领域 | 示例问题 |
|--------|----------|----------|----------|
| 1 | 民事纠纷 | 民事法律 | "邻居装修导致我家墙壁开裂，应该如何维权？" |
| 2 | 刑事案件 | 刑事法律 | "被指控盗窃但证据不足，应该如何辩护？" |
| 3 | 合同审查 | 合同法 | "签订房屋租赁合同需要注意哪些条款？" |
| 4 | 劳动纠纷 | 劳动法 | "公司无故辞退员工，应该怎么维权？" |
| 5 | 知识产权 | 知识产权法 | "如何申请软件著作权保护？" |
| 6 | 公司事务 | 公司法 | "有限责任公司和股份有限公司有什么区别？" |
| 7 | 婚姻家庭 | 婚姻家庭法 | "离婚时夫妻共同财产如何分割？" |
| 8 | 财产权益 | 物权法 | "购买二手房需要注意哪些法律问题？" |
| 9 | 行政争议 | 行政法 | "对行政处罚决定不服如何申请行政复议？" |
| 10 | 国际法律 | 国际法 | "跨国贸易合同适用哪国法律？" |

## 使用方式

### 交互式使用
```bash
python legal_assistant.py
```

### 命令行使用
```bash
# 单次问答
python legal_assistant.py -q "借款给朋友没有写借条，现在对方不还钱怎么办？" -s civil

# 批量处理
python legal_assistant.py -b sample_questions.json

# 指定模型路径
python legal_assistant.py --base-model Qwen/Qwen3-1.5B --lora-path ./lora_checkpoint
```

### 编程接口
```python
from legal_assistant import LawAssistant

# 创建助手实例
assistant = LawAssistant(
    base_model_path="Qwen/Qwen3-1.5B",
    lora_path="./output/lora_checkpoint"
)
assistant.load_model()

# 询问问题
response = assistant.ask_question(
    "公司无故辞退员工，应该怎么维权？", 
    scenario_type="labor"
)
print(response)
```

## 训练监控

项目集成SwanLab进行训练监控：
- 实时损失曲线
- 学习率变化
- 模型性能指标
- 训练日志记录
在终端登录自己的swanlab账号即可在云端观察训练曲线

## 模型性能

### 推理示例
```
Question: 公司无故辞退员工，应该怎么维权？

LLM:
<think>
用户询问公司无故辞退员工的维权方式。这是一个典型的劳动法问题。

首先需要分析法律依据：
1. 根据《劳动合同法》，用人单位单方解除劳动合同需要法定理由
2. 无故辞退属于违法解除劳动合同
3. 员工可以主张的权利包括：赔偿金、经济补偿金等

维权途径：
1. 劳动仲裁：首要选择，时效一年
2. 法院诉讼：对仲裁结果不服可起诉
3. 劳动监察：行政投诉途径

需要收集的证据：
- 劳动合同、工资流水
- 辞退通知书
- 工作证明、考勤记录
- 沟通记录等
</think>

根据《劳动合同法》相关规定，公司无故辞退员工属于违法解除劳动合同，您可以采取以下维权措施：

1. **收集证据**：包括劳动合同、工资银行流水、考勤记录、工作证、辞退通知书（如有）、与公司沟通的相关记录等。

2. **申请劳动仲裁**：这是最主要的维权途径。您需要在被辞退之日起一年内向用人单位所在地的劳动仲裁委员会申请仲裁，主张违法解除劳动合同赔偿金（通常为经济补偿金的二倍）。

3. **劳动监察投诉**：同时可以向当地劳动监察大队投诉，要求查处公司的违法行为。

4. **法院诉讼**：如对仲裁结果不服，可以在收到仲裁裁决书之日起15日内向人民法院提起诉讼。

建议您尽快采取行动，避免超过法定时效。如果需要具体的法律文书指导，建议咨询专业劳动法律师。
```

## 注意事项

### 医疗免责声明
- 本助手仅提供参考建议，不能替代专业法律咨询
- 紧急法律事务请立即咨询执业律师
- 所有建议仅供参考，具体案件请咨询专业律师

### 技术限制
- 模型基于训练数据，可能存在知识局限性
- 建议结合最新医学指南使用
- 定期更新模型以保持准确性

## 扩展开发

### 自定义模型路径
```python
assistant = LawAssistant(
    base_model_path="./your/base/model",
    lora_path="./your/lora/path",
    merged_model_path="./your/merged/model"
)
```

### 批量处理自定义问题
```json
[
  {
    "question": "您的问题",
    "scenario": "civil",
    "max_tokens": 512
  }
]
```

## 相关资源
- **数据集**: [CrimeKgAssitant](https://github.com/Liuyin159/CrimeKgAssitant)


