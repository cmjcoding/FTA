# download_bert_model.py
from transformers import BertForMaskedLM, BertTokenizerFast
# 在文件顶部添加（第2行）
from config import args  # 新增导入

model_config = {
    'ICEWS14': 'bert-base-cased',
    'ICEWS05': 'bert-base-cased',
    'ICEWS18': 'bert-large-cased'
}

model = BertForMaskedLM.from_pretrained(model_config[args.dataset])
tokenizer = BertTokenizerFast.from_pretrained(model_config[args.dataset])

save_path = f'./bert/bert_base_cased_{args.dataset}'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"模型已保存到 {save_path}")








tokenizer.save_pretrained('./bert/bert_base_cased_ICEWS14')

print("模型和Tokenizer已经保存到 ./bert/bert_base_cased_ICEWS14")
