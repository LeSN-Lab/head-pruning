import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import copy
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Subset
from torch.nn import CrossEntropyLoss, MSELoss
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.sentences = df.iloc[:, 1:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).values
        self.labels = df.iloc[:, 0].values - 1  # Label을 0부터 시작하도록 조정
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer(sentence, truncation=True, max_length=512, padding='max_length', return_tensors="pt")
        label = torch.tensor(self.labels[idx])

        return inputs, label

device = None

model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.to(device)

# test 데이터 가져오기
test_data = "./yahoo_answers_csv/test2.csv"
test_df = pd.read_csv(test_data)

test_dataset = TestDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# validation 데이터 가져오기
validation_data = "./yahoo_answers_csv/validation.csv"
validation_df = pd.read_csv(validation_data)

validation_dataset = TestDataset(validation_df, tokenizer)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

class Args:
    def __init__(self):
        self.device = device
        self.local_rank = -1  # 단일 GPU 사용을 가정
        self.output_mode = "classification"  # 또는 "regression"에 따라 설정
        self.num_labels = model.config.num_labels
        self.dont_normalize_importance_by_layer = False
        self.dont_normalize_global_importance = False

args = Args()

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print(device)
else:
    print("MPS device not found.")

def entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

per_class_importance_list = [torch.zeros(12, 12).to(args.device) for _ in range(10)]
per_class_token_list=[0.0 for _ in range(10)]
multihead_outputs_list = []  # 이 리스트에 각 layer의 출력을 저장합니다.

def hook_fn(module, input, output, layer_index):
    attention_value, attention_scores = output
    attention_value.requires_grad_(True)  # 그래디언트 계산을 위해 requires_grad를 True로 설정
    attention_value.retain_grad()
    multihead_outputs_list.append(attention_value)

def register_hooks(model):
    handles = []  # hook handles를 저장할 리스트
    for layer_index, layer in enumerate(model.bert.encoder.layer):
        handle = layer.attention.self.register_forward_hook(
            partial(hook_fn, layer_index=layer_index)
        )
        handles.append(handle)  # handle 저장
    return handles

def remove_hooks(handles):
    for handle in handles:
        handle.remove()

eval_dataloader = validation_loader  # 또는 validation_loader

# compute_entropy와 compute_importance는 True로 설정하여 기능을 활성화합니다.
compute_entropy = True
compute_importance = True

# 모든 헤드를 사용하여 importance score를 계산합니다.
head_mask = None

precision_list = [[0 for _ in range(11)] for _ in range(10)]
recall_list = [[0 for _ in range(11)] for _ in range(10)]
f1score_list = [[0 for _ in range(11)] for _ in range(10)]
global_accuracy_list = [[0 for _ in range(11)] for _ in range(10)]

total_precision_list = [[0 for _ in range(11)] for _ in range(10)]
total_recall_list = [[0 for _ in range(11)] for _ in range(10)]
total_f1score_list = [[0 for _ in range(11)] for _ in range(10)]
total_global_accuracy_list = [[0 for _ in range(11)] for _ in range(10)]

def compute_heads_importance(args, model, eval_dataloader, compute_entropy=True, compute_importance=True,
                             head_mask=None):
    # Prepare our tensors
    handles = register_hooks(model)
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    each_pred_head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)
    preds = None
    labels = None
    tot_tokens = 0.0

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        global count
        input_ids, label_ids = batch
        input_ids = {k: v.squeeze(1).to(device) for k, v in input_ids.items()}
        label_ids = label_ids.to(device)
        actual_batch_size = input_ids['input_ids'].size(0)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**input_ids, output_attentions=True)
        all_attentions = outputs[1]
        logits = outputs[0]

        if compute_entropy:
            # Update head attention entropy
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach())
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            # Update head importance scores with regards to our loss
            # First, backpropagate to populate the gradients
            if args.output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
            elif args.output_mode == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            loss.backward()

            # Second, compute importance scores according to http://arxiv.org/abs/1905.10650
            multihead_outputs = multihead_outputs_list
            for layer, mh_layer_output in enumerate(multihead_outputs):
                # print(layer)
                mh_layer_output_store = mh_layer_output
                reshaped_mh_layer_output = mh_layer_output_store.view(actual_batch_size, 512, 12, 64)
                reshaped_mh_layer_output = reshaped_mh_layer_output.permute(0, 2, 1, 3)

                mh_layer_output_grad = mh_layer_output.grad
                reshaped_mh_layer_output_grad = mh_layer_output_grad.view(actual_batch_size, 512, 12, 64)
                reshaped_mh_layer_output_grad = reshaped_mh_layer_output_grad.permute(0, 2, 1, 3)
                dot = torch.einsum("bhli,bhli->bhl", [reshaped_mh_layer_output_grad, reshaped_mh_layer_output])
                each_head_importance = dot.abs().sum(-1).sum(0).detach()
                head_importance[layer] += each_head_importance
                each_pred_head_importance[layer] += each_head_importance
            temp_each_pred_head_importance = copy.deepcopy(each_pred_head_importance)
            each_pred_head_importance.zero_()
            multihead_outputs_list.clear()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)
        prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)

        per_class_importance_list[prediction.item()] += temp_each_pred_head_importance
        per_class_token = (input_ids['input_ids'] != 0).float().sum().item()
        per_class_token_list[prediction.item()] = per_class_token
        tot_tokens += per_class_token

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    for i in range(10):
        per_class_importance_list[i] /= per_class_token_list[i]

    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        for i in range(10):
            norm_by_layer = torch.pow(torch.pow(per_class_importance_list[i], exponent).sum(-1), 1 / exponent)
            per_class_importance_list[i] /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())
        for i in range(10):
            per_class_importance_list[i] = (per_class_importance_list[i] - per_class_importance_list[i].min()) / (
                        per_class_importance_list[i].max() - per_class_importance_list[i].min())
    remove_hooks(handles)
    return attn_entropy, head_importance, preds, labels

def visualization_heatmap(file_name, array, num_layer, num_heads, label=0):
    # tensor를 CPU로 이동하여 numpy 배열로 변환
    array = array.cpu().numpy()

    df = pd.DataFrame(array)

    # 인덱스와 컬럼 이름 설정 (Layer와 Head의 인덱스를 1부터 시작하도록 조정)
    df.index = [f"Layer {i + 1}" for i in range(num_layer)]
    df.columns = [f"Head {i + 1}" for i in range(num_heads)]

    # 히트맵 생성 및 저장
    plt.figure(figsize=(12, 8))

    # Attention Score
    sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis')
    if label == 0 :
        plt.title("Total head important score")
    else:
        plt.title(f'Class {label} head important score')
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.savefig(f'./heatmap/head_importance_score/{file_name}.png')
    plt.close()

attn_entropy, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy, compute_importance, head_mask)

def calculate_prune_head(arr, i):
    # 2차원 배열 arr의 모든 요소와 해당 인덱스를 1차원 배열로 변환
    flattened_with_indices = [(value, index) for index, value in np.ndenumerate(arr)]

    # 값에 따라 오름차순 정렬하여 하위 12개 요소 선택
    sorted_by_value = sorted(flattened_with_indices, key=lambda x: x[0])
    bottom_12 = sorted_by_value[12 * i:12 * (i + 1)]

    # 하위 12개 요소의 인덱스만 추출
    bottom_12_indices = [index for _, index in bottom_12]

    return bottom_12_indices

for i in range(10):
    per_class_importance_list[i] = per_class_importance_list[i].cpu().numpy()

per_class_head_importance_list = copy.deepcopy(per_class_importance_list)

# layer별 max 값을 구함
def layer_max(arr):
    max_values = np.max(arr, axis=1)
    max_index = np.argmax(arr, axis=1)
    return max_index

def preprocess_prunehead(arr):
    for label in range(10):
        max_layer = layer_max(per_class_head_importance_list[label])
        for layer in range(12):
            head = max_layer[layer]
            per_class_head_importance_list[label][layer][head] = 100

def total_preprocess_prunehead(arr):
    max_layer = layer_max(arr)
    for layer in range(12):
            head = max_layer[layer]
            arr[layer][head] = 100
    return arr

def prune_head(model, prune_list):
    for layer_index, head_index in prune_list:
        model.bert.encoder.layer[layer_index].attention.prune_heads(([head_index]))
    return model

def print_prune_head_list(prune_list, trial):
    print(f"total prune number : {len(prune_list)*trial}")
    print(f"prune head list")
    print(prune_list)


def evaluating(model, class_index, prune_num):
    preds = []
    true_labels = []
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = batch
        inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1)

        preds.extend(prediction.tolist())
        true_labels.extend(labels.tolist())

    report = classification_report(true_labels, preds, output_dict=True)
    index = str(class_index)
    class_report = report[index]
    precision_list[class_index][prune_num] = class_report['precision']
    recall_list[class_index][prune_num] = class_report['recall']
    f1score_list[class_index][prune_num] = class_report['f1-score']
    global_accuracy_list[class_index][prune_num] = report['accuracy']

    print(f"Class {class_index} Precision: {class_report['precision']}")
    print(f"Class {class_index} Recall: {class_report['recall']}")
    print(f"Class {class_index} F1-Score: {class_report['f1-score']}")
    print(f"Global Accuracy: {report['accuracy']}")
    print()

def head_importance_prunning():
  for class_index in range(10):
      temp_model = copy.deepcopy(model)
      for num in range(11):
          print(f'Class {class_index+1} {(num+1)*12} prunning')
          prune_list = calculate_prune_head(per_class_head_importance_list[class_index], num)
          print_prune_head_list(prune_list, num+1)
          temp_model = prune_head(temp_model, prune_list)
          evaluating(temp_model, class_index, num)

head_importance_prunning()


def evaluating_all(model, prune_num):
    preds = []
    true_labels = []
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = batch
        inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1)

        preds.extend(prediction.tolist())
        true_labels.extend(labels.tolist())

    report = classification_report(true_labels, preds, output_dict=True)
    for i in range(10):
        index = str(i)
        class_report = report[index]
        total_precision_list[i][prune_num] = class_report['precision']
        total_recall_list[i][prune_num] = class_report['recall']
        total_f1score_list[i][prune_num] = class_report['f1-score']
        total_global_accuracy_list[i][prune_num] = report['accuracy']

        print(f"Class {i} Precision: {class_report['precision']}")
        print(f"Class {i} Recall: {class_report['recall']}")
        print(f"Class {i} F1-Score: {class_report['f1-score']}")
        print(f"Global Accuracy: {report['accuracy']}")
        print()

temp_head_importance_score = copy.deepcopy(head_importance).cpu().numpy()
temp_head_importance_score = total_preprocess_prunehead(temp_head_importance_score)

def total_head_importance_prunning():
    temp_model = copy.deepcopy(model)
    for num in range(11):
        print(f'Total {(num+1)*12} prunning')
        prune_list = calculate_prune_head(temp_head_importance_score, num)
        print_prune_head_list(prune_list, num+1)
        temp_model = prune_head(temp_model, prune_list)
        evaluating_all(temp_model, num)

total_head_importance_prunning()