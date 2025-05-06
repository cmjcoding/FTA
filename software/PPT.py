# PPT model
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
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from dataset import Dataset as DatasetFromMe
from utils import *
from joblib import Parallel, delayed
from torch.cuda.amp import autocast, GradScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')
rand_flag = False
if rand_flag:
    print('ramdom len')
else:
    print('fixed len')
dataset = args.dataset
max_sample = args.max_sample
seq_len = args.seq_len
ep = args.epoch
mi = args.mi
start_epoch = args.start_epoch
bert_path = './bert/bert_base_cased_{}'.format(dataset)
if start_epoch > 0:
    bert_path = './bert/bert_base_cased_{}/pretrain_{}/epoch_{}'.format(dataset, mi, start_epoch - 1)
    print('continue training from epoch {}'.format(start_epoch))
bert_path_test = './bert/bert_base_cased_{}/pretrain_{}/epoch_{}'.format(dataset, mi, ep)
mode = args.m
batch_size = args.batch_size
TOKENIZERS_PARALLELISM = (True | False)
superPrarams = {
    'ICEWS14': [28996, 7128, 36123],
    'ICEWS05': [28996, 10488, 39483],
    'ICEWS18': [28996, 23033, 52028]
}

token_begin = superPrarams[dataset][0]
token_sum = superPrarams[dataset][1]
token_end = superPrarams[dataset][2]

def sample_wrapper(quadruple, mode, dataTrain, dataValid, sample_num, rand_flag, dataset, numRel, id_rel, time_node, time_desp):
    idxSub = quadruple[0].item()
    quadruple = quadruple.tolist()

    filterQuad = dataTrain[dataTrain[:, 0] == idxSub]
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
                    self.time_desp
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
        bert_data = {'data': data, 'label': label, 'paths': paths}
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
        self.backbone_model = BertModel.from_pretrained(bert_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizerFast.from_pretrained('./bert/bert_base_cased_{}'.format(dataset))
        if mode == 'train':
            self.model = BertForMaskedLM.from_pretrained(bert_path)
        elif mode == 'viz':
            self.model = BertModel.from_pretrained(bert_path_test)
        else:
            self.model = BertForMaskedLM.from_pretrained(bert_path_test)
        # +++ 新增模糊类型模块 +++
        self.fuzzy_layer = FuzzyTypeAware(self.model.bert, self.model.config.hidden_size)

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
    def forward(self, input_ids, attention_mask, labels):
        with autocast():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

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
    rand_flag = True
    if pre_epochs > 0:
        bert_pre_train = md.get_data('train')
        pre_model = PreBert()
        pre_model.to(device)
        pre_model.init_tokenizer(entity_id)
        pre_model.init_new_token(entity_id)
        pre_optimizer = transformers.AdamW(pre_model.parameters(), lr=5e-5, correct_bias=True)

        # 创建 GradScaler 对象
        scaler = GradScaler()

        print('start training')
        pre_train_dataloader = pre_model.make_dataloader(bert_pre_train)
        pre_model.train()

        for epoch in range(pre_epochs):
            epoch_start_time = time.time()  # 记录每个epoch的开始时间
            total_loss = 0
            loop = tqdm(enumerate(pre_train_dataloader), total=len(pre_train_dataloader), leave=False)

            for step, batch in loop:
                batch_start_time = time.time()  # 记录每个batch的开始时间

                # 将数据移动到GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                text = batch['text']
                look_labels = labels.cpu().numpy()

                # 清零梯度
                pre_optimizer.zero_grad()

                # 使用 autocast 进行混合精度训练
                with autocast():
                    loss = pre_model(input_ids, attention_mask, labels)

                # 使用 scaler 来处理反向传播
                scaler.scale(loss).backward()
                scaler.step(pre_optimizer)
                scaler.update()

                total_loss += loss.item()

                # 计算batch处理时间并更新进度条
                batch_time = time.time() - batch_start_time
                loop.set_description(f'Epoch [{epoch + 1}/{pre_epochs}]')
                loop.set_postfix(loss=loss.item(), batch_time=f'{batch_time:.3f}s')

            # 计算和显示每个epoch的统计信息
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / len(pre_train_dataloader)
            print(f'Epoch {epoch + start_epoch + 1}/{pre_epochs + start_epoch}, '
                  f'Average Loss: {avg_loss:.4f}, '
                  f'Time: {epoch_time:.2f}s')

            # 保存模型
            save_path = './bert/bert_base_cased_{}/pretrain_{}/epoch_{}'.format(dataset, mi, epoch + start_epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pre_model.model.save_pretrained(save_path)

            # 打印GPU内存使用情况
            if torch.cuda.is_available():
                print(f"GPU memory after epoch: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")


# eval and test
def test():
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

    # 用 md.get_data 代替 get_test_data（避免多次 batch 并发采样）
    test_data = md.get_data('test')
    print(f"测试样本数量: {len(test_data['data'])}")

    # 初始化模型
    val_model = PreBert()
    val_model.to(device)
    val_model.init_tokenizer(entity_id)

    # 构建测试 DataLoader（只有一个 batch）
    test_dataset = val_model.make_dataloader(test_data)
    for batch in test_dataset:
        sample = batch
        break  # 只取第一个 batch 样本进行验证

    print("\n样本数据结构验证:")
    print(f"input_ids shape: {sample['input_ids'].shape} | 示例: {sample['input_ids'][0][:10]}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"tail_pos: {sample['tail_pos'][0]} | tail_index: {sample['tail_index'][0]}")
    print(f"文本示例: {sample['text'][0][:50]}...")

    print("\nForward过程验证:")
    dummy_input = sample['input_ids'][0].unsqueeze(0).to(device)
    dummy_mask = sample['attention_mask'][0].unsqueeze(0).to(device)
    dummy_labels = sample['labels'][0].unsqueeze(0).to(device)
    loss = val_model(dummy_input, dummy_mask, dummy_labels)
    print(f"Loss计算正常: {loss.item():.4f}")


def main():
    validate_data_pipeline()
    if mode == 'train':
        train(args.max_epochs)
    elif mode == 'test' or mode == 'eval':
        test()
    else:
        vis()

if __name__ == '__main__':
    main()

