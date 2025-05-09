# PPT model
# 1. 首先是所有的导入语句
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM, BertConfig
from dataset import Dataset as DatasetFromMe
from utils import *
from joblib import Parallel, delayed
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse
from make_data import MakeData
import torch
import numpy as np

# 2. 设置环境变量和警告

def parse_args():
    parser = argparse.ArgumentParser()

    # 添加默认值给关键的数值参数，并确保类型正确
    parser.add_argument('--vocab_size', type=int, default=0, help='词汇表大小')
    parser.add_argument('--dataset', type=str, default='ICEWS14', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--emb_size', type=int, default=768, help='嵌入维度')
    parser.add_argument('--max_epochs', type=int, default=1, help='最大训练轮数')
    parser.add_argument('--init', type=str, default='', help='初始化方法')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU设备 ID (默认: -1, 使用 CPU)')
    parser.add_argument('--max_sample', type=int, default=50, help='最大采样数')
    parser.add_argument('--seq_len', type=int, default=128, help='序列长度')
    parser.add_argument('--m', type=str, default='train', help='模式')
    parser.add_argument('--epoch', type=int, default=0, help='训练轮数')
    parser.add_argument('--mi', type=int, default=0, help='mi参数')
    parser.add_argument('--entity_epoch', type=int, default=0, help='实体训练轮数')
    parser.add_argument('--rel_epoch', type=int, default=0, help='关系训练轮数')
    parser.add_argument('--start_epoch', type=int, default=0, help='开始轮数')
    parser.add_argument('--test_sample', type=int, default=50, help='测试样本数')
    parser.add_argument('--rand_flag', type=int, default=0, help='随机标志')
    parser.add_argument('--pre_epochs', type=int, default=1, help='预训练轮数')
    parser.add_argument('--finetune_epochs', type=int, default=1, help='微调轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--second', type=int, default=0, help='second参数')
    parser.add_argument('--fuzzy_threshold', type=float, default=0.0, help='模糊阈值')
    parser.add_argument('--type_embed_size', type=int, default=768, help='类型嵌入维度')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    parser.add_argument('--debug_level', choices=['DEBUG', 'INFO', 'WARNING'], default='DEBUG', help='调试日志级别')
    parser.add_argument('--ablation', choices=['none', 'no_fuzzy', 'no_context'], default='none',
                        help='控制消融实验：none表示使用模糊嵌入，no_fuzzy禁用模糊融合，no_context只用原始嵌入')

    args = parser.parse_args()

    # 设置环境变量和警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings('ignore')

    # 设置设备
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu)
        print('GPU{} used'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('CPU used')

    print('dataset: {} used'.format(args.dataset))

    # 初始化数据集
    D = DatasetFromMe(args.dataset)
    id_entity, entity_id = D.get_entity_by_id()
    args.entity_num = len(entity_id)

    # 确保所有可能用于比较的数值参数都是正确的类型
    numeric_args = [
        'vocab_size', 'batch_size', 'emb_size', 'max_epochs', 'max_sample',
        'seq_len', 'epoch', 'mi', 'entity_epoch', 'rel_epoch', 'start_epoch',
        'test_sample', 'rand_flag', 'pre_epochs', 'finetune_epochs', 'seed',
        'second', 'type_embed_size'
    ]

    for arg_name in numeric_args:
        if hasattr(args, arg_name):
            try:
                setattr(args, arg_name, int(getattr(args, arg_name)))
            except (ValueError, TypeError):
                setattr(args, arg_name, 0)

    # 确保浮点数参数类型正确
    float_args = ['lr', 'fuzzy_threshold']
    for arg_name in float_args:
        if hasattr(args, arg_name):
            try:
                setattr(args, arg_name, float(getattr(args, arg_name)))
            except (ValueError, TypeError):
                setattr(args, arg_name, 0.0)

    # 添加其他必要参数
    args.output_dir = './checkpoints'
    args.save_steps = 100
    args.device = device

    return args


def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

    # 解析参数


args = parse_args()

# 设置超参数
superPrarams = {
    'ICEWS14': [28996, 7128, 36123],
    'ICEWS05': [28996, 10488, 39483],
    'ICEWS18': [28996, 23033, 52028]
}

token_begin = superPrarams[args.dataset][0]
token_sum = superPrarams[args.dataset][1]
token_end = superPrarams[args.dataset][2]

# 设置模型路径
bert_path = f'./bert/bert_base_cased_{args.dataset}'
if args.start_epoch > 0:
    bert_path = f'./bert/bert_base_cased_{args.dataset}/pretrain_{args.mi}/epoch_{args.start_epoch - 1}'
    print(f'continue training from epoch {args.start_epoch}')

bert_path_test = f'./bert/bert_base_cased_{args.dataset}/pretrain_{args.mi}/epoch_{args.epoch}'

# 6. 其他配置
TOKENIZERS_PARALLELISM = (True | False)

superPrarams = {
    'ICEWS14': [28996, 7128, 36123],
    'ICEWS05': [28996, 10488, 39483],
    'ICEWS18': [28996, 23033, 52028]
}

token_begin = superPrarams[dataset][0]
token_sum = superPrarams[dataset][1]
token_end = superPrarams[dataset][2]
def sample_wrapper(quadruple, mode, dataTrain, dataValid, sample_num, rand_flag, dataset, numRel, id_rel, time_node, time_desp, strategy='head'):
    idxSub = quadruple[0].item()
    quadruple = quadruple.tolist()

    if strategy == 'head':
        filterQuad = dataTrain[dataTrain[:, 0] == quadruple[0]]
    elif strategy == 'rel':
        filterQuad = dataTrain[dataTrain[:, 1] == quadruple[1]]
    elif strategy == 'tail':
        filterQuad = dataTrain[dataTrain[:, 2] == quadruple[2]]
    else:
        filterQuad = dataTrain[dataTrain[:, 0] == quadruple[0]]  # 默认


    if mode == 'test':
        filterQuad_valid = dataValid[dataValid[:, 0] == idxSub]
        filterQuad = torch.cat((filterQuad, filterQuad_valid), dim=0)

    if mode == 'train':
        if not rand_flag:
            sampleQuad = filterQuad[torch.randperm(filterQuad.shape[0])][:sample_num]
        else:
            sample_num = random.randint(2, sample_num)
            sampleQuad = filterQuad[torch.randperm(filterQuad.shape[0])][:sample_num]
    else:
        if dataset == 'ICEWS05':
            filterQuad = filterQuad[filterQuad[:, 3] < quadruple[3]]
        sampleQuad = filterQuad[torch.randperm(filterQuad.shape[0])][:sample_num - 1]
        sampleQuad = torch.cat((sampleQuad, torch.tensor(quadruple).unsqueeze(0)), dim=0)

    sampleQuad = sampleQuad[torch.argsort(sampleQuad[:, 3])]
    sampleQuad[1:, 3] = sampleQuad[1:, 3] - sampleQuad[:-1, 3]
    sampleQuad[0, 3] = -1

    # 构造句子
    sentence = []
    path = []
    for i, quad in enumerate(sampleQuad):
        head = f'[ENT{2 * i}]'
        rel = id_rel[int(quad[1])]
        tail = f'[ENT{2 * i + 1}]'
        time = "Unknown"
        for j in range(1, len(time_node)):
            if int(quad[3]) < time_node[j]:
                time = time_desp[j - 1]
                break
        else:
            time = time_desp[-1]
        text = f"{time} {head} {rel} {tail}"
        sentence.append(text)
        path.append(quad.tolist())

    final_text = '[TRI] ' + ' [TRI] '.join(sentence)
    return final_text, int(path[-1][2]), path


class MakeData:
    def __init__(self, dataset, sample_num, rand_flag):
        self.dataset = dataset
        self.D = DatasetFromMe(dataset)
        self.time_node, self.time_desp = self.get_time_dict()
        self.numEnt, self.numRel = self.D.get_num()
        self.id_entity, self.entity_id = self.D.get_entity_by_id()
        self.id_rel, self.rel_id = self.D.get_rel_by_id()
        stamp, max_time = self.D.get_stamp_and_max_time()
        self.sample_num = sample_num
        self.rand_flag = rand_flag
        self.dataTrain = torch.LongTensor(self.D.data_for_dynamic_train())
        self.dataValid = torch.LongTensor(self.D.data_for_dynamic_valid())
        self.dataTest = torch.LongTensor(self.D.data_for_dynamic_test())
        self.dataTrain[:, 3] = self.dataTrain[:, 3] // stamp
        self.dataValid[:, 3] = self.dataValid[:, 3] // stamp
        self.dataTest[:, 3] = self.dataTest[:, 3] // stamp
        self.dataTrain_ = self.dataTrain[:, [2, 1, 0, 3]]
        self.dataTrain_[:, 1] = self.dataTrain_[:, 1] + self.numRel
        self.dataValid_ = self.dataValid[:, [2, 1, 0, 3]]
        self.dataValid_[:, 1] = self.dataValid_[:, 1] + self.numRel
        self.dataTest_ = self.dataTest[:, [2, 1, 0, 3]]
        self.dataTest_[:, 1] = self.dataTest_[:, 1] + self.numRel
        self.dataTrain = torch.cat((self.dataTrain, self.dataTrain_), dim=0)
        self.dataValid = torch.cat((self.dataValid, self.dataValid_), dim=0)
        self.dataTest = torch.cat((self.dataTest, self.dataTest_), dim=0)

    def get_time_dict(self):
        with open('./data/{}/time_dict.txt'.format(self.dataset), 'r') as f:
            rows = f.readlines()
            time_node = []
            time_desp = []
            for row in rows:
                seq = row.replace('\n', '').split(':')
                time_node.append(int(seq[0]))
                time_desp.append(seq[1])
            return time_node, time_desp

    def get_entity_type(self, entity_id):
        # 简单示例：返回一个默认类型，比如 'Entity'
        return "Entity"

    def invert_time_desp(self, time_interval):
        for i in range(1, len(self.time_node)):
            if time_interval < self.time_node[i]:
                return self.time_desp[i - 1]
        return self.time_desp[-1]

    def sample(self, quadruple, mode):
        idxSub = quadruple[0]
        filterQuad = self.dataTrain[self.dataTrain[:, 0] == idxSub][:, :]
        if mode == 'test':
            filterQuad_valid = self.dataValid[self.dataValid[:, 0] == idxSub][:, :]
            filterQuad = torch.cat((filterQuad, filterQuad_valid), dim=0)
        if mode == 'train':
            if not self.rand_flag:
                sampleQuad = filterQuad[torch.randperm(filterQuad.shape[0])][:self.sample_num]
            else:
                sample_num = random.randint(2, self.sample_num)
                sampleQuad = filterQuad[torch.randperm(filterQuad.shape[0])][:sample_num]
        else:
            if args.dataset == 'ICEWS05':
                filterQuad = filterQuad[filterQuad[:, 3] < quadruple[3]]
            sampleQuad = filterQuad[torch.randperm(filterQuad.shape[0])][:self.sample_num - 1]
            sampleQuad = torch.cat((sampleQuad, quadruple.unsqueeze(0)), dim=0)
        sampleQuad = sampleQuad[torch.argsort(sampleQuad[:, 3])]
        sampleQuad[1:, 3] = sampleQuad[1:, 3] - sampleQuad[:-1, 3]
        sampleQuad[0, 3] = -1
        text, path = self.quad2sentence(sampleQuad)
        return text, int(path[-1][2]), path

    def quad2sentence(self, x):
        sentence = []
        path = []
        i = 0
        for quadruple in x:
            sentence.append(self.quad2sentence_single(quadruple, i))
            path.append(quadruple)
            i += 1
        text = ' [TRI] '.join(sentence)
        text = '[TRI] ' + text
        return text, path

    @lru_cache(maxsize=1024)
    def quad2sentence_single(self, quadruple_tuple, i):
        quadruple = torch.tensor(quadruple_tuple)
        head = '[ENT{}]'.format(2 * i)
        rel = self.id_rel[int(quadruple[1])]
        tail = '[ENT{}]'.format(2 * i + 1)
        tim = self.invert_time_desp(int(quadruple[3]))
        return tim + ' ' + head + ' ' + rel + ' ' + tail

    def batch_sample(self, mode):
        if mode == 'train':
            data = self.dataTrain
        elif mode == 'valid':
            data = self.dataValid
        else:
            data = self.dataTest

        results = []
        print(f"[DEBUG] 开始串行采样，共 {len(data[:50])} 个样本")  # 可加日志观察
        for i, quad in enumerate(data[:50]):  # 仅前50个样本用于测试加速
            try:
                r = sample_wrapper(
                    quad, mode,
                    self.dataTrain,
                    self.dataValid,
                    self.sample_num,
                    self.rand_flag,
                    self.dataset,
                    self.numRel,
                    self.id_rel,
                    self.time_node,
                    self.time_desp,
                    strategy='head'  # 可切换为'rel' 或 'tail'
                )

                results.append(r)
            except Exception as e:
                print(f"[ERROR] 第 {i} 个样本采样失败：{e}")
                continue

        if len(results) == 0:
            print("[ERROR] 无有效采样结果，停止执行")
            exit()

        texts = [r[0] for r in results]
        labels = [r[1] for r in results]
        paths = [r[2] for r in results]
        return texts, torch.LongTensor(labels), paths

    def get_data(self, mode):
        data, label, paths = self.batch_sample(mode)
        # 确保返回字典格式
        bert_data = {
            'data': data,          # 这应该是一个列表
            'label': label,        # 这应该是一个张量
            'paths': paths         # 这应该是一个列表
        }
        print(f"[DEBUG] get_data 返回数据结构:")
        print(f"- data type: {type(bert_data['data'])}")
        print(f"- label type: {type(bert_data['label'])}")
        print(f"- paths type: {type(bert_data['paths'])}")
        return bert_data

    def get_test_data(self, mode):
        test_batchs = []
        for i in range(args.test_sample):
            data, label, paths = self.batch_sample(mode)
            bert_data = {'data': data, 'label': label, 'paths': paths}
            test_batchs.append(bert_data)
        return test_batchs


class TestBertDataset:
    def __init__(self, tokenizer, max_length, bert_data_batchs):
        super(TestBertDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bert_data_batchs = bert_data_batchs

    def __len__(self):
        return len(self.bert_data_batchs[0]['data'])

    def __getitem__(self, batch_id, index):
        text = self.bert_data_batchs[batch_id]['data'][index]
        path = self.bert_data_batchs[batch_id]['paths'][index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].flatten()
        input_ids_mask = input_ids.detach().clone()
        for i, id in enumerate(input_ids):
            id = int(id - 1)
            if 0 <= id < 33:
                row = id // 2
                col = (id % 2) * 2
                input_ids[i] = path[row][col] + token_begin
        tail_mask = input_ids.detach().clone()
        labels = input_ids.detach().clone()
        last_non_zero = torch.nonzero(labels, as_tuple=False)[-1]
        tail_pos = last_non_zero - 1
        tail_index = tail_mask[tail_pos]
        tail_mask[tail_pos] = 103
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': inputs['attention_mask'].flatten(),
            'tail_mask': tail_mask,
            'tail_index': tail_index,
            'tail_pos': tail_pos,
            'text': text
        }


class FuzzyTypeAware(nn.Module):
    def __init__(self, bert_model, entity_dim):
        super().__init__()
        self.bert = bert_model
        self.proj = nn.Linear(bert_model.config.hidden_size, entity_dim)

    def get_context_rep(self, masked_input):
        with torch.no_grad():
            outputs = self.bert(**masked_input)
            return outputs.last_hidden_state[:, 0]  # 拿[CLS]

    def compute_membership(self, entity_emb, context_emb):
        # 计算模糊隶属度 μ ∈ [0,1]
        cosine_sim = F.cosine_similarity(entity_emb, context_emb, dim=-1)
        return (cosine_sim + 1) / 2  # 归一化到[0,1]



class PreBertDataset(Dataset):
    def __init__(self, tokenizer, max_length, bert_data):
        super(PreBertDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = bert_data['data']
        self.label = bert_data['label']
        self.paths = bert_data['paths']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        path = self.paths[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].flatten()
        input_ids_mask = input_ids.detach().clone()
        for i, id in enumerate(input_ids):
            id = int(id - 1)
            if 0 <= id < 33:
                row = id // 2
                col = (id % 2) * 2
                input_ids[i] = path[row][col] + token_begin
        tail_mask = input_ids.detach().clone()
        labels = input_ids.detach().clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.3) * (input_ids != 101) * (input_ids != 102) * (input_ids != 103)
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        input_ids[selection] = 103
        last_non_zero = torch.nonzero(labels, as_tuple=False)[-1]
        tail_pos = last_non_zero - 1
        tail_index = tail_mask[tail_pos]
        tail_mask[tail_pos] = 103

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': inputs['attention_mask'].flatten(),
            'tail_mask': tail_mask,
            'tail_index': tail_index,
            'tail_pos': tail_pos,
            'text': self.data[index]
        }


class PreBert(nn.Module):
    def __init__(self):
        super(PreBert, self).__init__()

        # 1. 设置调试模式和日志
        self.debug_mode = args.debug
        self.logger = self._setup_logger()

        try:
            # 2. 初始化tokenizer
            self.tokenizer = BertTokenizerFast.from_pretrained(
                f'./bert/bert_base_cased_{dataset}'
            )

            # 3. 获取模型配置并设置正确的词汇表大小
            config = BertConfig.from_pretrained(bert_path)
            # 确保词汇表大小正确（使用实际观察到的最大值）
            max_token_id = max(token_end, 31166)  # 31165 + 1 为安全起见
            self.vocab_size = max_token_id
            config.vocab_size = self.vocab_size

            if self.debug_mode:
                self.logger.debug(f"Vocabulary size adjustment:")
                self.logger.debug(f"- Original vocab size: {config.vocab_size}")
                self.logger.debug(f"- New vocab size: {self.vocab_size}")

            # 4. 初始化模型
            model_init_kwargs = {
                'config': config,
                'ignore_mismatched_sizes': True
            }

            self.backbone_model = BertModel.from_pretrained(bert_path, **model_init_kwargs)
            self.mlm_model = BertForMaskedLM.from_pretrained(bert_path, **model_init_kwargs)

            if mode == 'train':
                self.model = BertForMaskedLM.from_pretrained(bert_path, **model_init_kwargs)
            elif mode == 'viz':
                self.model = BertModel.from_pretrained(bert_path_test, **model_init_kwargs)
            else:
                self.model = BertForMaskedLM.from_pretrained(bert_path_test, **model_init_kwargs)

            # 5. 初始化模糊类型模块
            self.fuzzy_layer = FuzzyTypeAware(self.model.bert, config.hidden_size)

            # 6. 保存配置
            self.config = self.model.config

            if self.debug_mode:
                self.logger.debug("Model initialization completed successfully")
                self.logger.debug(f"Model configuration: {self.config}")

        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            if self.debug_mode:
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(logging.INFO)
        return logger

    def _log_batch_stats(self, loss_value, batch_idx, optimizer):
        """记录批次训练统计信息"""
        if self.debug_mode:
            current_lr = optimizer.param_groups[0]['lr']
            grad_norm = 0.0
            param_norm = 0.0

            # 计算梯度和参数范数
            for p in self.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
                    param_norm += p.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            param_norm = param_norm ** 0.5

            self.logger.debug(f"Batch {batch_idx} statistics:")
            self.logger.debug(f"- Loss: {loss_value:.4f}")
            self.logger.debug(f"- Learning rate: {current_lr:.6f}")
            self.logger.debug(f"- Gradient norm: {grad_norm:.4f}")
            self.logger.debug(f"- Parameter norm: {param_norm:.4f}")
            self.logger.debug(f"- GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB")
            self.logger.debug(f"- GPU Memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f}MB")

            # 记录每层的梯度统计信息
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    self.logger.debug(
                        f"- Layer {name}: "
                        f"grad_mean={grad.mean().item():.4f}, "
                        f"grad_std={grad.std().item():.4f}, "
                        f"grad_max={grad.abs().max().item():.4f}"
                    )

    def _setup_logger(self):
        # 设置日志记录器
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        return logger

    def init_tokenizer(self, entity_id):
        if not os.path.exists('./bert/bert_base_cased_{}'.format(dataset) + '/added_tokens.json'):
            print('add new words to bert')
            new_tokens = []
            for i in range(len(entity_id)):
                new_tokens.append('ENT-' + str(i))
            self.tokenizer.add_tokens(new_tokens)
            self.backbone_model.resize_token_embeddings(len(self.tokenizer))
            self.mlm_model.resize_token_embeddings(len(self.tokenizer))
            self.fuzzy_layer.bert.resize_token_embeddings(len(self.tokenizer))  # ✅ 添加这一行
            self.tokenizer.save_pretrained('./bert/bert_base_cased_{}'.format(dataset))
        else:
            print('load tokenizer')
            self.backbone_model.resize_token_embeddings(len(self.tokenizer))
            self.mlm_model.resize_token_embeddings(len(self.tokenizer))
            self.fuzzy_layer.bert.resize_token_embeddings(len(self.tokenizer))  # ✅ 添加这一行

    def init_new_token(self, entity_id):
        for entity in entity_id:
            tokenized_entity = self.tokenizer.tokenize(entity)
            tokenized_entity = self.tokenizer.convert_tokens_to_ids(tokenized_entity)
            tokenized_entity = self.model.bert.embeddings.word_embeddings.weight.data[tokenized_entity]
            entity_embedding = torch.mean(tokenized_entity, dim=0)
            self.model.bert.embeddings.word_embeddings.weight.data[-7128 + entity_id[entity]] = entity_embedding

    def make_dataloader(self, bert_data):
        template_bert_dataset = PreBertDataset(self.tokenizer, seq_len, bert_data)
        template_bert_dataloader = DataLoader(template_bert_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        return template_bert_dataloader

    def make_test_data(self, bert_data_batchs):
        bert_batchs = TestBertDataset(self.tokenizer, seq_len, bert_data_batchs)
        return bert_batchs


# +++ 替换forward函数 +++
    def forward(self, input_ids, attention_mask, labels=None):
        """
            前向传播方法
            Args:
                input_ids: 输入token的ID
                attention_mask: 注意力掩码
                labels: 标签（可选，用于训练）
            """

        if args.ablation != 'no_fuzzy':
            try:
                # 创建 masked_input 用于上下文建模（CLS聚合）
                masked_input = {
                    "input_ids": input_ids.clone(),
                    "attention_mask": attention_mask.clone()
                }
                masked_input['input_ids'][:, 1] = 103  # 通常把第一个实体替换为[MASK]
                masked_input = {k: v.to(self.model.device) for k, v in masked_input.items()}

                # 获取上下文嵌入
                context_emb = self.fuzzy_layer.get_context_rep(masked_input)  # (batch, hidden)
                origin_emb = self.model.bert.embeddings.word_embeddings(input_ids[:, 1])  # 实体原始嵌入 (batch, hidden)

                # 计算 μ 并融合
                mu = self.fuzzy_layer.compute_membership(origin_emb, context_emb).unsqueeze(-1)  # (batch, 1)
                if args.ablation == 'no_context':
                    fused_emb = origin_emb
                else:
                    fused_emb = mu * context_emb + (1 - mu) * origin_emb  # (batch, hidden)

                # 替换第一个[MASK]实体的嵌入
                input_embeds = self.model.bert.embeddings(input_ids)
                input_embeds[:, 1, :] = fused_emb  # 用融合后的嵌入替换原嵌入
            except Exception as e:
                if self.debug_mode:
                    self.logger.error(f"模糊类型嵌入融合失败: {str(e)}")
                input_embeds = None
        else:
            input_embeds = None


        if self.debug_mode:
            self.logger.debug("Forward pass started")
            self.logger.debug(f"Input shapes:")
            self.logger.debug(f"- input_ids: {input_ids.shape}")
            self.logger.debug(f"- attention_mask: {attention_mask.shape}")
            self.logger.debug(f"- labels: {labels.shape}")
            if labels is not None:
                self.logger.debug(f"- labels: {labels.shape}")

            try:
                # 确保输入在正确的设备上
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
                if labels is not None:
                    labels = labels.to(self.model.device)

                # 模型前向传播
                outputs = self.model(
                    inputs_embeds=input_embeds if input_embeds is not None else None,
                    input_ids=None if input_embeds is not None else input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                if self.debug_mode:
                    self.logger.debug("Forward pass completed")
                    if hasattr(outputs, 'loss'):
                        self.logger.debug(f"Loss: {outputs.loss.item():.4f}")

                # 如果是训练模式，返回loss
                if labels is not None:
                    return outputs.loss
                # 如果是推理模式，返回预测结果
                return outputs.logits

            except Exception as e:
                if self.debug_mode:
                    self.logger.error(f"Forward pass error: {str(e)}")
                    if 'outputs' in locals():
                        self.logger.error(f"Outputs shape: {outputs.logits.shape}")
                raise

    def _validate_inputs(self, input_ids, attention_mask, labels):
        """验证输入数据的形状和值"""
        if self.debug_mode:
            assert input_ids.dim() == 2, f"输入维度应为2维,实际为: {input_ids.shape}"
            assert attention_mask.shape == input_ids.shape, \
                f"注意力掩码形状 {attention_mask.shape} 与输入形状不匹配 {input_ids.shape}"

            # 检查数值范围
            self.logger.debug(f"Input value ranges:")
            self.logger.debug(f"- input_ids: [{input_ids.min()}, {input_ids.max()}]")
            self.logger.debug(f"- attention_mask: [{attention_mask.min()}, {attention_mask.max()}]")

    def _log_tensor_stats(self, name, tensor):
        """记录张量的统计信息"""
        if self.debug_mode:
            self.logger.debug(f"{name} statistics:")
            self.logger.debug(f"- shape: {tensor.shape}")
            self.logger.debug(f"- dtype: {tensor.dtype}")
            self.logger.debug(f"- device: {tensor.device}")
            self.logger.debug(f"- mean: {tensor.float().mean().item():.4f}")
            self.logger.debug(f"- std: {tensor.float().std().item():.4f}")

    def result(self, tail_mask, attention_mask, tail_index, text, tail_pos, labels):
        outputs = self.model(tail_mask, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        # 使用向量化操作替代循环
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        tail_logits = logits[batch_indices, tail_pos.squeeze(), :]
        return tail_logits[:, -token_sum:]

    def viz(self, tail_mask, attention_mask, labels, tail_pos):
        outputs = self.model(tail_mask, attention_mask=attention_mask, labels=labels)
        attention = outputs[-1]
        tokens = self.tokenizer.convert_ids_to_tokens(tail_mask[0].cpu().numpy())
        logits = outputs.logits
        tail_logits = torch.zeros((logits.shape[0], logits.shape[2])).cuda()
        for i in range(logits.shape[0]):
            tail_logits[i, :] = logits[i, tail_pos[i, 0], :]
        _, pred = torch.max(tail_logits, dim=1)
        print(pred)
        return attention, tokens, pred


warnings.filterwarnings(action='ignore')
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
    torch.cuda.set_device(args.gpu)
    print('GPU{} used'.format(args.gpu))
else:
    device = torch.device('cpu')
    print('CPU used')
print('dataset: {} used'.format(args.dataset))
D = DatasetFromMe(dataset)
id_entity, entity_id = D.get_entity_by_id()
entity_num = len(entity_id)
md = MakeData(dataset, max_sample, rand_flag)


# train
def train(pre_epochs):
    print('start training')

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    bert_pre_train = md.get_data('train')

    # 创建模型并设置debug模式
    pre_model = PreBert()
    if args.debug:
        pre_model.debug_mode = True
        pre_model.logger.setLevel(getattr(logging, args.debug_level))

    pre_model.to(device)
    pre_model.train()  # 设置为训练模式

    # 初始化优化器
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.lr)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )

    # 创建数据加载器
    train_dataloader = pre_model.make_dataloader(bert_pre_train)

    # 用于记录最佳模型
    best_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    # 创建日志文件
    log_file = os.path.join(args.output_dir, 'training_log.txt')

    # 创建进度条
    pbar = tqdm(total=pre_epochs * len(train_dataloader), desc='Training')

    try:
        for epoch in range(pre_epochs):
            if pre_model.debug_mode:
                pre_model.logger.debug(f"\nEpoch {epoch + 1}/{pre_epochs} started")

            epoch_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # 将数据移动到设备
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    # 清零梯度
                    optimizer.zero_grad()

                    # 前向传播
                    outputs = pre_model(input_ids, attention_mask, labels)
                    loss = outputs  # forward方法已经返回loss

                    # 检查损失值是否为NaN
                    if torch.isnan(loss):
                        raise ValueError("Loss is NaN")

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(pre_model.parameters(), max_norm=1.0)

                    optimizer.step()

                    # 记录损失
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    num_batches += 1

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f'{epoch + 1}/{pre_epochs}',
                        'batch': f'{batch_idx}/{len(train_dataloader)}',
                        'loss': f'{current_loss:.4f}'
                    })

                    # 记录批次统计信息
                    if pre_model.debug_mode:
                        pre_model._log_batch_stats(current_loss, batch_idx, optimizer)

                    # 每N个批次保存检查点
                    if batch_idx > 0 and batch_idx % args.save_steps == 0:
                        checkpoint_path = os.path.join(
                            args.output_dir,
                            f'checkpoint_e{epoch}_b{batch_idx}.pt'
                        )
                        torch.save({
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'model_state_dict': pre_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': current_loss,
                        }, checkpoint_path)

                        # 记录到日志文件
                        with open(log_file, 'a') as f:
                            f.write(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {current_loss:.4f}\n")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        pre_model.logger.error(f"GPU OOM in batch {batch_idx}. Skipping batch.")
                        continue
                    raise e

            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss / num_batches

            # 更新学习率调度器
            scheduler.step(avg_epoch_loss)

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(pre_model.state_dict(), best_model_path)
                print(f"New best model saved with loss: {best_loss:.4f}")

            # 记录epoch结果到日志文件
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss:.4f}\n")

            print(f'Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss:.4f}')

    except Exception as e:
        if pre_model.debug_mode:
            pre_model.logger.error(f"Training error: {str(e)}")
        raise

    finally:
        pbar.close()

    return pre_model, best_loss

# eval and test
def test():
    print('start test/eval epoch{}'.format(ep))
    print('now is {}'.format(args.mode))
    bert_test_batchs = md.get_test_data(args.mode)

    # 创建模型并设置debug模式
    eval_model = PreBert()
    if args.debug:
        eval_model.debug_mode = True
        eval_model.logger.setLevel(getattr(logging, args.debug_level))

    eval_model.to(device)
    error_count = 0
    print('start test/eval epoch{}'.format(ep))
    print('now is {}'.format(args.mode))
    bert_test_batchs = md.get_test_data(args.mode)
    hitsRank = [1, 3, 10]
    mrrCount = 0
    hitsCount = [0 for i in range(len(hitsRank))]
    eval_model = PreBert()
    eval_model.to(device)
    eval_model.init_tokenizer(entity_id)
    test_data = eval_model.make_test_data(bert_test_batchs)
    eval_model.eval()
    with torch.no_grad():
        examplesLen = len(bert_test_batchs[0]['data'])
        for idx in trange(examplesLen):
            scores = []
            for batch_num in range(args.test_sample):
                single_test = test_data.__getitem__(batch_num, idx)
                tail_mask = single_test['tail_mask'].to(device)
                attention_mask = single_test['attention_mask'].to(device)
                tail_index = single_test['tail_index'].to(device)
                labels = single_test['labels'].to(device)
                text = single_test['text']
                tail_pos = single_test['tail_pos']
                tail_mask = tail_mask.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                tail_index = tail_index.unsqueeze(0)
                labels = labels.unsqueeze(0)
                text = [text]
                tail_pos = tail_pos.unsqueeze(0)
                score = eval_model.result(tail_mask, attention_mask, tail_index, text, tail_pos, labels)
                scores.append(score)
                tail_index = token_sum - 1 - torch.abs(tail_index - token_end)
            score = torch.mean(torch.stack(scores), dim=0)
            tail_index[tail_index < 0] = 0
            mrrC, hitsC = calc_mrr_count(score, tail_index, hitsRank)
            mrrCount += mrrC
            for i in range(len(hitsRank)):
                hitsCount[i] += hitsC[i]
        print('count error: {}'.format(error_count))
        mrr = mrrCount / examplesLen
        print('mrr: {:.2f}'.format(mrr * 100))
        results = ['{:.2f}'.format(mrr * 100)]
        for i in range(len(hitsRank)):
            print('hit@{}: {:.2f}'.format(hitsRank[i], hitsCount[i] / examplesLen * 100))
            results.append('{:.2f}'.format(hitsCount[i] / examplesLen * 100))
        print('\t'.join(results))


def vis():
    bert_test_batchs = md.get_test_data('test')
    eval_model = PreBert()
    if args.debug:
        eval_model.debug_mode = True
        eval_model.logger.setLevel(getattr(logging, args.debug_level))
    eval_model.to(device)
    eval_model.init_tokenizer(entity_id)
    test_data = eval_model.make_test_data(bert_test_batchs)
    eval_model.eval()
    with torch.no_grad():
        examplesLen = len(bert_test_batchs[0]['data'])
        for idx in trange(examplesLen):
            single_test = test_data.__getitem__(0, idx)
            tail_mask = single_test['tail_mask'].to(device)
            attention_mask = single_test['attention_mask'].to(device)
            tail_index = single_test['tail_index'].to(device)
            labels = single_test['labels'].to(device)
            text = single_test['text']
            tail_pos = single_test['tail_pos']
            tail_mask = tail_mask.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            tail_index = tail_index.unsqueeze(0)
            labels = labels.unsqueeze(0)
            text = [text]
            tail_pos = tail_pos.unsqueeze(0)
            input_id_list = tail_mask.tolist()[0]
            tokens = eval_model.tokenizer.convert_ids_to_tokens(input_id_list)
            for i in range(len(tokens)):
                if tokens[i] == '[TRI]':
                    tokens[i] = '[EVE]'
            result = eval_model.model(tail_mask, output_attentions=True)
            attention = result[-1]
            last_attention = attention[-1]
            last_attention = last_attention.squeeze(0)
            last_attention = last_attention[:, :tail_pos + 1, :tail_pos + 1]
            last_attention = torch.mean(last_attention, dim=0)
            last_attention = last_attention.detach().cpu().numpy()
            mask_attention = last_attention[:, tail_pos].squeeze()
            max_value = np.max(mask_attention)
            if tail_pos < 36:
                draw_input = np.zeros((6, 6))
                for i in range(6):
                    for j in range(6):
                        ind = i * 6 + j
                        if ind < len(mask_attention):
                            draw_input[i, j] = mask_attention[ind]
                fig = sns.heatmap(draw_input,
                                  linewidth=0.5,
                                  fmt='',
                                  vmax=max_value,
                                  vmin=0,
                                  cmap="YlGnBu")
                for i in range(6):
                    for j in range(6):
                        ind = i * 6 + j
                        if ind < len(tokens):
                            plt.text(j + 0.5, i + 0.5, tokens[ind], ha='center', va='center')
                plt.title(str(tail_index[0][0]))
                plt.savefig('./vis/{}.pdf'.format(idx))
                plt.close()


# ...（前面原有的类定义代码保持不变）...

# 数据预处理验证模块（新增部分）
def validate_data_pipeline():
    print("\n=== 开始数据预处理验证 ===")
    try:
        test_data = md.get_data('test')
        print(f"获取到的测试数据类型: {type(test_data)}")

        if isinstance(test_data, dict):
            print(f"测试样本数量: {len(test_data.get('data', []))}")

            # 创建验证模型
            val_model = PreBert()
            val_model.to(device)

            # 获取一个小批量数据
            batch_size = min(args.batch_size, 4)  # 使用较小的批量进行验证
            sample_data = {
                'data': test_data['data'][:batch_size],
                'label': test_data['label'][:batch_size] if isinstance(test_data['label'], torch.Tensor) else test_data[
                    'label'],
                'paths': test_data['paths'][:batch_size]
            }

            # 准备输入数据
            dummy_input = torch.randint(0, val_model.vocab_size - 1, (batch_size, 128)).to(device)
            dummy_mask = torch.ones((batch_size, 128)).to(device)
            dummy_labels = torch.randint(0, val_model.vocab_size - 1, (batch_size, 128)).to(device)

            print("\n测试随机输入:")
            loss = val_model(dummy_input, dummy_mask, dummy_labels)
            print(f"随机输入测试成功，损失值: {loss.item():.4f}")

            print("\n测试实际数据:")
            test_dataset = val_model.make_dataloader(sample_data)
            for batch in test_dataset:
                loss = val_model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['labels'].to(device)
                )
                print(f"实际数据测试成功，损失值: {loss.item():.4f}")
                break

    except Exception as e:
        print(f"数据预处理验证失败: {str(e)}")
        print("\n详细的数据结构:")
        if 'test_data' in locals():
            print(f"test_data类型: {type(test_data)}")
            if isinstance(test_data, dict):
                print("字典键:", list(test_data.keys()))
        raise


def main():
    try:
        # 首先进行数据预处理验证
        validate_data_pipeline()

        if mode == 'train':
            # 在train函数内部设置debug模式
            train(args.max_epochs)
        elif mode == 'test' or mode == 'eval':
            # 在test函数内部设置debug模式
            test()
        else:
            # 在vis函数内部设置debug模式
            vis()
    except Exception as e:
        print(f"运行出错: {str(e)}")
        raise
    print(f"运行模式：{args.m} | 消融设定：{args.ablation}")


if __name__ == '__main__':
    main()

