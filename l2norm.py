import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import Transformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import copy
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

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print(device)
else:
    print("MPS device not found.")

# 모델 및 토크나이저 로드
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

# 결과를 저장할 딕셔너리 초기화
init_dict = lambda: {'attention_score_sum': 0, 'attention_score_l2norm_sum': 0, 'attention_distribution':0, 'attention_value_sum': 0, 'attention_value_l2norm_sum': 0}
results = {label: {layer: {head: init_dict() for head in range(model.config.num_attention_heads)} for layer in range(model.config.num_hidden_layers)} for label in range(model.config.num_labels)}

temp_result = {layer: {head: init_dict() for head in range(model.config.num_attention_heads)} for layer in range(model.config.num_hidden_layers)}
temp_result_list = []

# 3차원 배열 초기화
attention_scores_l2norm_averages = np.zeros((model.config.num_labels, model.config.num_hidden_layers, model.config.num_attention_heads))
attention_values_l2norm_averages = np.zeros((model.config.num_labels, model.config.num_hidden_layers, model.config.num_attention_heads))

correct_prediction = [0] * model.config.num_labels
class_total = [0] * model.config.num_labels
total_predictions = 0
total_correct_predictions = 0

score_precision_list = [[0 for _ in range(11)] for _ in range(10)]
score_recall_list = [[0 for _ in range(11)] for _ in range(10)]
score_f1score_list = [[0 for _ in range(11)] for _ in range(10)]
score_global_accuracy_list = [[0 for _ in range(11)] for _ in range(10)]

value_precision_list = [[0 for _ in range(11)] for _ in range(10)]
value_recall_list = [[0 for _ in range(11)] for _ in range(10)]
value_f1score_list = [[0 for _ in range(11)] for _ in range(10)]
value_global_accuracy_list = [[0 for _ in range(11)] for _ in range(10)]

def hook_fn(module, input, output, layer_index):
    attention_value_concate = output[0]
    attention_value_per_head = attention_value_concate[0].chunk(12, dim=1)
    attention_scores = output[1]
    # 각 head에 대해 L2 Norm 계산 및 합계 업데이트를 진행
    for head_index in range(model.config.num_attention_heads):
        attention_score_per_head = attention_scores[0][head_index]
        attention_scores_l2norm = torch.norm(attention_score_per_head, p=2)
        attention_values_l2norm = torch.norm(attention_score_per_head[head_index], p=2)

        temp_result[layer_index][head_index]['attention_score_sum'] = attention_score_per_head
        temp_result[layer_index][head_index]['attention_score_l2norm_sum'] = attention_scores_l2norm.item()
        temp_result[layer_index][head_index]['attention_value_sum'] = attention_value_per_head[head_index]
        temp_result[layer_index][head_index]['attention_value_l2norm_sum'] = attention_values_l2norm.item()

    if layer_index == model.config.num_hidden_layers - 1:
        temp_result_copy = copy.deepcopy(temp_result)
        temp_result_list.append(temp_result_copy)

def register_hooks(model):
    handles = []  # hook handles를 저장할 리스트
    for layer_index, layer in enumerate(model.bert.encoder.layer):
        handle = layer.attention.self.register_forward_hook(
            lambda module, input, output, layer_index=layer_index: hook_fn(module, input, output, layer_index)
        )
        handles.append(handle)  # handle 저장
    return handles

def remove_hooks(handles):
    for handle in handles:
        handle.remove()

handles = register_hooks(model)

with torch.no_grad():
    for inputs, label in tqdm(validation_loader, desc="Evaluating"):
        inputs = {key: value.squeeze(1).to(device) for key, value in inputs.items()}
        label = label.to(device)

        # 모델의 출력을 구합니다.
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        # 예측이 정확한지 여부를 계산
        correct = predictions == label

        # 배치 내의 각 샘플에 대해 반복
        for idx in range(label.size(0)):
            total_predictions += 1
            class_total[label[idx].item()] += 1
            if correct[idx]:
                total_correct_predictions += 1
                correct_prediction[label[idx].item()] += 1
                for layer in range(12):
                    for head in range(12):
                        results[label[idx].item()][layer][head]['attention_score_sum'] += temp_result_list[0][layer][head]['attention_score_sum']
                        results[label[idx].item()][layer][head]['attention_score_l2norm_sum'] += temp_result_list[0][layer][head]['attention_score_l2norm_sum']
                        results[label[idx].item()][layer][head]['attention_value_sum'] += temp_result_list[0][layer][head]['attention_value_sum']
                        results[label[idx].item()][layer][head]['attention_value_l2norm_sum'] += temp_result_list[0][layer][head]['attention_value_l2norm_sum']
            temp_result_list.pop(0)

for label in range(10):
    for layer in range(12):
        for head in range(12):
            attention_scores_l2norm_averages[label][layer][head] = results[label][layer][head]['attention_score_l2norm_sum'] / correct_prediction[label]
            attention_values_l2norm_averages[label][layer][head] = results[label][layer][head]['attention_value_l2norm_sum'] / correct_prediction[label]

def print_accuracy():
    total_accuracy = (total_correct_predictions / total_predictions) * 100
    print(f"Total Accuracy: {total_accuracy:.2f}%")
    for i in range (0, 10):
        print(f"Accuracy of class {i} : {correct_prediction[i] / class_total[i]*100}%")
print_accuracy()

# layer별 max 값을 구함
def layer_max(arr):
    max_values = np.max(arr, axis=1)
    max_index = np.argmax(arr, axis=1)
    return max_index

def preprocess_prunehead(arr):
    for label in range(10):
        max_layer = layer_max(arr[label])
        for layer in range(12):
            head = max_layer[layer]
            arr[label][layer][head] = 1000

def calculate_prune_head(arr, i):
    # 2차원 배열 arr의 모든 요소와 해당 인덱스를 1차원 배열로 변환
    flattened_with_indices = [(value, index) for index, value in np.ndenumerate(arr)]

    # 값에 따라 오름차순 정렬하여 하위 12개 요소 선택
    sorted_by_value = sorted(flattened_with_indices, key=lambda x: x[0])
    bottom_12 = sorted_by_value[12 * i:12 * (i + 1)]

    # 하위 12개 요소의 인덱스만 추출
    bottom_12_indices = [index for _, index in bottom_12]

    return bottom_12_indices

def prune_head(model, prune_list):
    for layer_index, head_index in prune_list:
        model.bert.encoder.layer[layer_index].attention.prune_heads(([head_index]))
    return model

def print_prune_head_list(prune_list, trial):
    print(f"total prune number : {len(prune_list)*trial}")
    print(f"prune head list")
    print(prune_list)


def evaluating_score(model, class_index, prune_num):
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

    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    index = str(class_index)
    class_report = report[index]
    score_precision_list[class_index][prune_num] = class_report['precision']
    score_recall_list[class_index][prune_num] = class_report['recall']
    score_f1score_list[class_index][prune_num] = class_report['f1-score']
    score_global_accuracy_list[class_index][prune_num] = report['accuracy']

    print(f"Class {class_index + 1} Precision: {class_report['precision']}")
    print(f"Class {class_index + 1} Recall: {class_report['recall']}")
    print(f"Class {class_index + 1} F1-Score: {class_report['f1-score']}")
    print(f"Global Accuracy: {report['accuracy']}")
    print()


def evaluating_value(model, class_index, prune_num):
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

    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    index = str(class_index)
    class_report = report[index]
    value_precision_list[class_index][prune_num] = class_report['precision']
    value_recall_list[class_index][prune_num] = class_report['recall']
    value_f1score_list[class_index][prune_num] = class_report['f1-score']
    value_global_accuracy_list[class_index][prune_num] = report['accuracy']

    print(f"Class {class_index + 1} Precision: {class_report['precision']}")
    print(f"Class {class_index + 1} Recall: {class_report['recall']}")
    print(f"Class {class_index + 1} F1-Score: {class_report['f1-score']}")
    print(f"Global Accuracy: {report['accuracy']}")
    print()

def score_prunning():
  for class_index in range(10):
      temp_model = copy.deepcopy(model)
      for num in range(11):
          print(f'Class {class_index+1} {(num+1)*12} prunning')
          prune_list = calculate_prune_head(attention_scores_l2norm_averages[class_index], num)
          print_prune_head_list(prune_list, num+1)
          temp_model = prune_head(temp_model, prune_list)
          evaluating_score(temp_model, class_index, num)

def value_prunning():
  for class_index in range(10):
      temp_model = copy.deepcopy(model)
      for num in range(11):
          print(f'Class {class_index+1} {(num+1)*12} prunning')
          prune_list = calculate_prune_head(attention_values_l2norm_averages[class_index], num)
          print_prune_head_list(prune_list, num+1)
          temp_model = prune_head(temp_model, prune_list)
          evaluating_value(temp_model, class_index, num)

remove_hooks(handles)

test_attention_scores = copy.deepcopy(attention_scores_l2norm_averages)
test_attention_values = copy.deepcopy(attention_values_l2norm_averages)
for i in range(10):
    preprocess_prunehead(test_attention_scores)
    preprocess_prunehead(test_attention_values)

score_prunning()
value_prunning()