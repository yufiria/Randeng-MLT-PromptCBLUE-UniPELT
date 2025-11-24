#!/usr/bin/env python
# coding=utf-8

"""
使用UniPELT方法对Baichuan-13B-Chat模型进行参数高效微调。
支持PromptCBLUE医疗数据集，一键运行完整训练流程。
"""

import os
import sys
import json
import logging

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

# 添加项目路径
sys.path.append("./")

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_and_process_data(data_file, tokenizer, max_length=512):
    """
    加载和处理训练数据
    """
    logger.info(f"加载数据文件: {data_file}")
    
    # 读取JSONL文件
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"加载了 {len(data)} 条训练数据")
    
    # 处理数据
    processed_data = []
    for item in data:
        input_text = item.get('input', '')
        target_text = item.get('target', '')
        
        # 构建对话格式，Baichuan模型专用
        prompt = f"<reserved_106>{input_text}<reserved_107>{target_text}"
        
        # Tokenize
        tokenized = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # 设置labels用于loss计算
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        processed_data.append(tokenized)
    
    # 转换为datasets格式
    from datasets import Dataset
    dataset = Dataset.from_list(processed_data)
    
    return dataset


def main():
    """
    主训练函数
    """
    
    logger.info("="*60)
    logger.info("开始Baichuan-13B-Chat + UniPELT微调训练")
    logger.info("="*60)
    
    # 配置参数
    model_name = "baichuan-inc/Baichuan-13B-Chat"
    output_dir = "checkpoint/baichuan-unipelt"
    train_file = "datasets/PromptCBLUE/aug_train_verb.json"
    
    # LoRA配置
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    
    # 训练配置
    num_train_epochs = 3
    per_device_train_batch_size = 4  # Baichuan-13B需要更多显存
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    max_length = 512
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"模型: {model_name}")
    logger.info(f"训练数据: {train_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"LoRA配置: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # 加载分词器
    logger.info("加载分词器...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    logger.info("加载Baichuan-13B-Chat模型...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    logger.info(f"原始模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 应用LoRA配置
    logger.info("应用LoRA配置...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["W_pack"],
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # 加载数据
    logger.info("加载和处理训练数据...")
    
    if not os.path.exists(train_file):
        logger.error(f"训练数据文件不存在: {train_file}")
        logger.info("请确保数据文件在正确的位置")
        return
    
    train_dataset = load_and_process_data(train_file, tokenizer, max_length)
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    # 初始化Trainer
    logger.info("初始化Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练...")
    logger.info("="*60)
    
    train_result = trainer.train()
    
    # 保存模型
    logger.info("保存模型...")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 保存配置信息
    config_info = {
        "base_model": model_name,
        "peft_type": "LoRA",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": f"{100 * trainable_params / all_params:.2f}%",
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    logger.info("="*60)
    logger.info("训练完成！")
    logger.info(f"模型保存在: {output_dir}")
    logger.info(f"可训练参数比例: {100 * trainable_params / all_params:.2f}%")
    logger.info("="*60)
    
    # 使用说明
    logger.info("\n使用训练好的模型进行推理：")
    logger.info("```python")
    logger.info("from peft import PeftModel")
    logger.info("from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info("")
    logger.info(f"# 加载基础模型")
    logger.info(f"base_model = AutoModelForCausalLM.from_pretrained('{model_name}', trust_remote_code=True)")
    logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{model_name}', trust_remote_code=True)")
    logger.info("")
    logger.info(f"# 加载LoRA adapter")
    logger.info(f"model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    logger.info("")
    logger.info("# 进行推理")
    logger.info("prompt = '<reserved_106>文本分类任务：判断以下句子的意图<reserved_107>'")
    logger.info("inputs = tokenizer(prompt, return_tensors='pt')")
    logger.info("outputs = model.generate(**inputs, max_new_tokens=100)")
    logger.info("print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
    logger.info("```")


if __name__ == "__main__":
    # 检查PEFT是否安装
    try:
        import peft
        logger.info(f"PEFT版本: {peft.__version__}")
    except ImportError:
        print("错误：未安装PEFT库")
        print("请运行: pip install peft")
        sys.exit(1)
    
    # 检查训练数据是否存在
    train_file = "datasets/PromptCBLUE/aug_train_verb.json"
    if not os.path.exists(train_file):
        print(f"警告：训练数据文件不存在: {train_file}")
        print("请确保数据文件在正确的位置")
    
    # 运行主函数
    main()
