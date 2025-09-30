"""
法律助手集成脚本
基于 Qwen3-1.7B + LoRA 微调的法律模型，提供多种法律场景的智能助手功能
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig  # 新增Peft相关导入
import argparse
import json
import time
from datetime import datetime
import os

# 法律专业提示词模板（保持不变）
LAW_PROMPTS = {
    "civil": "你是一名经验丰富的民事律师，请根据案件描述，提供专业的法律分析和维权建议。",
    "criminal": "你是一名刑事辩护律师，请分析案件情况，提供专业的法律意见和辩护策略。",
    "contract": "你是一名合同法律专家，请审查合同条款，提供专业的法律风险评估和修改建议。",
    "labor": "你是一名劳动法律师，请根据劳动纠纷情况，提供专业的权益保护建议和法律途径。",
    "intellectual_property": "你是一名知识产权律师，请提供专业的版权、专利、商标保护建议。",
    "corporate": "你是一名公司法律师，请就公司设立、运营、治理等提供专业法律意见。",
    "family": "你是一名婚姻家庭法律师，请就婚姻、继承、抚养等问题提供专业法律建议。",
    "property": "你是一名物权法律专家，请就房产、土地等财产权利提供专业法律分析。",
    "administrative": "你是一名行政法律师，请就行政处罚、行政许可等提供专业法律意见。",
    "international": "你是一名国际法律师，请就跨国法律事务提供专业建议。"
}

# 常见法律场景（保持不变）
LAW_SCENARIOS = {
    "1": "民事纠纷",
    "2": "刑事案件", 
    "3": "合同审查",
    "4": "劳动纠纷",
    "5": "知识产权",
    "6": "公司事务",
    "7": "婚姻家庭",
    "8": "财产权益",
    "9": "行政争议",
    "10": "国际法律"
}

# 预设问题示例（保持不变）
SAMPLE_QUESTIONS = {
    "civil": [
        "邻居装修导致我家墙壁开裂，应该如何维权？",
        "借款给朋友没有写借条，现在对方不还钱怎么办？",
        "网购商品存在质量问题，商家拒绝退货该如何处理？"
    ],
    "criminal": [
        "被指控盗窃但证据不足，应该如何辩护？",
        "正当防卫的认定标准是什么？",
        "刑事案件中如何申请取保候审？"
    ],
    "contract": [
        "签订房屋租赁合同需要注意哪些条款？",
        "劳动合同中的竞业限制条款是否有效？",
        "如何审查投资协议中的风险条款？"
    ],
    "labor": [
        "公司无故辞退员工，应该怎么维权？",
        "加班费应该如何计算和主张？",
        "工伤认定需要提供哪些材料？"
    ],
    "intellectual_property": [
        "如何申请软件著作权保护？",
        "发现他人侵权使用我的商标该怎么办？",
        "专利被侵权应该如何收集证据？"
    ],
    "corporate": [
        "有限责任公司和股份有限公司有什么区别？",
        "公司股东之间发生纠纷如何解决？",
        "公司并购需要注意哪些法律风险？"
    ],
    "family": [
        "离婚时夫妻共同财产如何分割？",
        "遗嘱继承和法定继承有什么区别？",
        "子女抚养权的判定标准是什么？"
    ],
    "property": [
        "购买二手房需要注意哪些法律问题？",
        "房屋拆迁补偿标准如何确定？",
        "小区公共区域的权益归属如何界定？"
    ],
    "administrative": [
        "对行政处罚决定不服如何申请行政复议？",
        "行政许可被拒绝应该怎么办？",
        "政府信息公开申请被驳回如何救济？"
    ],
    "international": [
        "跨国贸易合同适用哪国法律？",
        "在海外投资需要注意哪些法律风险？",
        "国际仲裁与诉讼有什么区别？"
    ]
}

class LawAssistant:
    def __init__(self, base_model_path=None, lora_path=None, merged_model_path=None):
        """
        初始化法律助手
        
        Args:
            base_model_path: 基础模型路径
            lora_path: LoRA适配器路径
            merged_model_path: 合并后的模型路径（如果已合并）
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.merged_model_path = merged_model_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def _select_device_and_dtype(self):
        """选择设备和数据类型"""
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32
    
    def load_model(self):
        """加载模型和分词器 - 支持LoRA微调模型"""
        print("正在加载法律助手模型...")
        
        # 优先级1: 使用合并后的模型（最优选择）
        if self.merged_model_path and os.path.exists(self.merged_model_path):
            print("🔧 加载合并后的模型...")
            self._load_merged_model()
        # 优先级2: 使用基础模型 + LoRA适配器
        elif self.base_model_path and self.lora_path and os.path.exists(self.base_model_path) and os.path.exists(self.lora_path):
            print("🔧 加载基础模型 + LoRA适配器...")
            self._load_lora_model()
        # 优先级3: 回退到只使用基础模型
        elif self.base_model_path and os.path.exists(self.base_model_path):
            print("⚠️ 使用回退模式：仅加载基础模型（无LoRA）...")
            self._load_base_model_only()
        else:
            raise ValueError("""
请提供有效的模型路径：
1. 合并模型路径 (--merged-model)
2. 基础模型路径 + LoRA路径 (--base-model 和 --lora-path)  
3. 仅基础模型路径 (--base-model)
            """)
        
        print(f"模型加载完成！使用设备: {self.device}")
    
    def _load_merged_model(self):
        """加载合并后的模型"""
        # 检查路径是否存在
        if not os.path.exists(self.merged_model_path):
            raise FileNotFoundError(f"合并模型路径不存在: {self.merged_model_path}")
        
        print(f"📥 从 {self.merged_model_path} 加载合并模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.merged_model_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.merged_model_path, 
            torch_dtype=self.dtype,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ 合并模型加载成功")
    
    def _load_lora_model(self):
        """加载基础模型 + LoRA适配器"""
        # 检查路径是否存在
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"基础模型路径不存在: {self.base_model_path}")
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"LoRA适配器路径不存在: {self.lora_path}")
        
        print(f"📥 从 {self.base_model_path} 加载基础模型...")
        print(f"📥 从 {self.lora_path} 加载LoRA适配器...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, 
            torch_dtype=self.dtype,
            local_files_only=True
        )
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            base_model,
            self.lora_path,
            torch_dtype=self.dtype,
            local_files_only=True
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # 打印可训练参数信息
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ LoRA模型加载成功")
        print(f"📊 模型参数: 可训练 {trainable_params:,} / 总计 {total_params:,}")
    
    def _load_base_model_only(self):
        """回退模式：仅加载基础模型"""
        # 检查路径是否存在
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"基础模型路径不存在: {self.base_model_path}")
        
        print(f"📥 从 {self.base_model_path} 加载基础模型（无LoRA）...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, 
            torch_dtype=self.dtype,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ 基础模型加载成功（未使用LoRA微调）")
        print("💡 提示：要使用LoRA微调效果，请提供 --lora-path 参数")
    
    def predict(self, messages, max_new_tokens=512):
        """执行预测"""
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # 添加采样以获得更自然的输出
            temperature=0.7,  # 控制创造性
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 只解码新生成部分
        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response
    
    # 以下方法保持不变...
    def ask_question(self, question, scenario_type="civil", max_tokens=512):
        """询问法律问题"""
        if scenario_type not in LAW_PROMPTS:
            scenario_type = "civil"
        
        messages = [
            {"role": "system", "content": LAW_PROMPTS[scenario_type]},
            {"role": "user", "content": question}
        ]
        
        # 记录对话历史
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_type,
            "question": question,
            "response": None
        })
        
        response = self.predict(messages, max_new_tokens=max_tokens)
        
        # 更新对话历史
        self.conversation_history[-1]["response"] = response
        
        return response
    
    def show_scenarios(self):
        """显示可用的法律场景"""
        print("\n⚖️ 法律助手 - 可用场景:")
        print("=" * 50)
        for key, value in LAW_SCENARIOS.items():
            print(f"{key:2}. {value}")
        print("=" * 50)
    
    def show_sample_questions(self, scenario_type):
        """显示示例问题"""
        if scenario_type in SAMPLE_QUESTIONS:
            print(f"\n📋 {LAW_SCENARIOS.get(scenario_type, '法律咨询')} - 示例问题:")
            print("-" * 40)
            for i, question in enumerate(SAMPLE_QUESTIONS[scenario_type], 1):
                print(f"{i}. {question}")
            print("-" * 40)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🤖 法律助手已启动！")
        print("输入 'help' 查看帮助，输入 'quit' 退出")
        
        while True:
            try:
                # 显示场景选择
                self.show_scenarios()
                
                # 选择场景
                scenario_choice = input("\n请选择法律场景 (1-10): ").strip()
                if scenario_choice == 'quit':
                    break
                elif scenario_choice == 'help':
                    self.show_help()
                    continue
                elif scenario_choice not in LAW_SCENARIOS:
                    print("❌ 无效选择，请重新输入")
                    continue
                
                # 获取场景类型
                scenario_type = list(LAW_PROMPTS.keys())[int(scenario_choice) - 1]
                
                # 显示示例问题
                self.show_sample_questions(scenario_type)
                
                # 获取用户问题
                question = input(f"\n请输入您的{LAW_SCENARIOS[scenario_choice]}问题: ").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue
                
                # 生成回答
                print("\n🔄 正在分析您的法律问题...")
                start_time = time.time()
                
                response = self.ask_question(question, scenario_type)
                
                end_time = time.time()
                
                # 显示回答
                elapsed_time = end_time - start_time
                print(f"\n💡 法律助手回答 (耗时: {elapsed_time:.2f}秒):")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
                # 询问是否继续
                continue_choice = input("\n是否继续咨询？(y/n): ").strip().lower()
                if continue_choice in ['n', 'no', '否']:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用法律助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 法律助手使用帮助:")
        print("=" * 50)
        print("1. 选择法律场景 (1-10)")
        print("2. 输入您的法律问题")
        print("3. 获得专业的法律建议")
        print("\n💡 重要提示:")
        print("- 本助手仅提供法律知识参考，不构成正式法律意见")
        print("- 具体案件请咨询执业律师")
        print("- 输入 'quit' 退出程序")
        print("=" * 50)
    
    def save_conversation(self, filename=None):
        """保存对话历史"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"law_conversation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"💾 对话历史已保存到: {filename}")
    
    def batch_questions(self, questions_file):
        """批量处理问题"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            print(f"📝 开始批量处理 {len(questions)} 个法律问题...")
            
            results = []
            for i, q in enumerate(questions, 1):
                print(f"\n处理第 {i}/{len(questions)} 个问题...")
                response = self.ask_question(
                    q.get('question', ''), 
                    q.get('scenario', 'civil'),
                    q.get('max_tokens', 512)
                )
                
                results.append({
                    "question": q.get('question', ''),
                    "scenario": q.get('scenario', 'civil'),
                    "response": response
                })
            
            # 保存结果
            output_file = f"law_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 批量处理完成！结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="法律助手 - 基于Qwen3 + LoRA微调的智能法律咨询系统")
    
    # 模型路径参数组
    model_group = parser.add_argument_group('模型路径设置')
    model_group.add_argument("--base-model", type=str, 
                           default="Qwen/Qwen3-1.7B",
                           help="基础模型路径或名称")
    model_group.add_argument("--lora-path", type=str, 
                           help="LoRA适配器路径")
    model_group.add_argument("--merged-model", type=str, 
                           help="合并后的模型路径")
    
    # 功能参数
    parser.add_argument("--question", "-q", type=str, 
                       help="直接询问问题（需要配合 --scenario 使用）")
    parser.add_argument("--scenario", "-s", type=str, 
                       default="civil", 
                       choices=list(LAW_PROMPTS.keys()),
                       help="法律场景类型")
    parser.add_argument("--max-tokens", "-m", type=int, 
                       default=512, 
                       help="最大生成token数")
    parser.add_argument("--batch", "-b", type=str, 
                       help="批量处理问题文件（JSON格式）")
    parser.add_argument("--save-history", action="store_true", 
                       help="保存对话历史")
    
    args = parser.parse_args()
    
    # 创建法律助手实例
    assistant = LawAssistant(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        merged_model_path=args.merged_model
    )
    
    # 加载模型
    assistant.load_model()
    
    if args.batch:
        # 批量处理模式
        assistant.batch_questions(args.batch)
    elif args.question:
        # 单次问答模式
        print(f"🤖 法律助手回答:")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        # 交互模式
        assistant.interactive_mode()
    
    # 保存对话历史
    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()