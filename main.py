import re
import random
import time
from statistics import mode
import wandb
import torch.optim as optim
import os
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # 既存の処理に加えて
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    return text.strip()

class JapaneseBERTEncoder:
    def __init__(self, model_name='cl-tohoku/bert-base-japanese-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, sentences, device='cuda'):
        self.model = self.model.to(device) 
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # 入力をデバイスに移動
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output.last_hidden_state[:, 0, :]  # CLS token


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer

        # HuggingFaceのclass_mappingを使用
        self.class_mapping = pandas.read_csv('https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv')
        self.answer2idx = {row['answer']: row['class_id'] for _, row in self.class_mapping.iterrows()}
        self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        # 質問文の処理は不要になります（BERTが処理するため）

    def update_dict(self, dataset):

        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = process_text(self.df["question"][idx])  # 前処理のみ行い、エンコードはしない

        if self.answer:
            answers = [self.answer2idx.get(process_text(answer["answer"]), -1) for answer in self.df["answers"][idx]]
            answers = [a for a in answers if a != -1]  # 未知の回答を除外
            mode_answer_idx = mode(answers) if answers else -1

            return image, question, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, question

    def __len__(self):
        return len(self.df)

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
# def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
#     total_acc = 0.

#     for pred, answers in zip(batch_pred, batch_answers):
#         acc = 0.
#         for i in range(len(answers)):
#             num_match = 0
#             for j in range(len(answers)):
#                 if i == j:
#                     continue
#                 if pred == answers[j]:
#                     num_match += 1
#             acc += min(num_match / 3, 1)
#         total_acc += acc / 10

#     return total_acc / len(batch_pred)




# ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        self.resnet = ResNet50()
        self.text_encoder = JapaneseBERTEncoder()
        
        sentence_embedding_dim = 768
        image_feature_dim = 512  # ResNet50の最後の全結合層の出力次元
        
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + sentence_embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(512, n_answer)

    def forward(self, image, question):
        image_feature = self.resnet(image)
        question_feature = self.text_encoder.encode(question, device=image.device)
        
        # 画像特徴量と質問特徴量を結合
        combined = torch.cat([image_feature, question_feature], dim=1)
        
        # 特徴量の融合
        fused = self.fusion(combined)
        
        # 分類
        logits = self.classifier(fused)

        return logits 

def custom_loss(pred, target, answers):
    # パディングされた部分（-1）をマスク
    mask = (answers != -1).float()
    
    # クロスエントロピー損失を計算
    loss = F.cross_entropy(pred, target, reduction='none')
    
    # マスクを適用して、パディングされた部分の損失を0にする
    masked_loss = loss * mask.any(dim=1).float()
    
    # 有効な回答の平均損失を返す
    return masked_loss.sum() / mask.any(dim=1).sum()

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for i, (image, question, answers, mode_answer) in enumerate(dataloader):
        if i % 200 == 0:
            print(f"{i} / {len(dataloader)}")
        image, answers, mode_answer = image.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = custom_loss(pred, mode_answer, answers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


# def eval(model, dataloader, device):
#     model.eval()
#     total_loss = 0
#     total_acc = 0
#     simple_acc = 0

#     start = time.time()
#     with torch.no_grad():
#         for batch in dataloader:
#             if len(batch) == 4:  # 訓練データの場合
#                 image, question, answers, mode_answer = batch
#                 image, answers, mode_answer = image.to(device), answers.to(device), mode_answer.to(device)

#                 pred = model(image, question)
#                 loss = custom_loss(pred, mode_answer, answers)

#                 total_loss += loss.item()
#                 total_acc += VQA_criterion(pred.argmax(1), answers)
#                 simple_acc += (pred.argmax(1) == mode_answer).mean().item()
#             else:  # テストデータの場合
#                 image, question = batch
#                 image = image.to(device)

#                 pred = model(image, question)

#     if len(batch) == 4:  # 訓練データの場合のみ平均を計算
#         num_batches = len(dataloader)
#         return total_loss / num_batches, total_acc / num_batches, simple_acc / num_batches, time.time() - start
#     else:  # テストデータの場合
#         return 0, 0, 0, time.time() - start

def eval(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    total_samples = 0
    start = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 4:  # 訓練/検証データの形式（画像、質問、回答、最頻値の回答）
                image, question, answers, mode_answer = batch
                image, answers, mode_answer = image.to(device), answers.to(device), mode_answer.to(device)
                
                pred = model(image, question)
                loss = custom_loss(pred, mode_answer, answers)
                
                total_loss += loss.item() * len(image)
                total_acc += VQA_criterion(pred.argmax(1), answers) * len(image)
                simple_acc += (pred.argmax(1) == mode_answer).float().sum().item()
                total_samples += len(image)
            else:  
                image, question = batch
                image = image.to(device)
                
                pred = model(image, question)
                total_samples += len(image)

    eval_time = time.time() - start
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_simple_acc = simple_acc / total_samples
        return avg_loss, avg_acc, avg_simple_acc, eval_time
    else:
        return 0, 0, 0, eval_time
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    batch_size = batch_pred.size(0)

    for pred, answers in zip(batch_pred, batch_answers):
        valid_answers = answers[answers != -1]  # パディングを除外
        if len(valid_answers) == 0:
            continue  # 有効な回答がない場合はスキップ

        acc = 0.
        num_match = (pred == valid_answers).sum().item()
        acc = min(num_match / 3, 1)
        total_acc += acc

    return total_acc / batch_size

def custom_collate(batch):
    if len(batch[0]) == 4:  # 訓練データの場合
        images, questions, answers, mode_answers = zip(*batch)
        
        images = torch.stack(images, 0)
        questions = list(questions)
        
        max_len = max(len(a) for a in answers)
        padded_answers = [torch.cat([a, torch.full((max_len - len(a),), -1, dtype=a.dtype)]) for a in answers]
        answers = torch.stack(padded_answers)
        
        mode_answers = torch.tensor(mode_answers)
        
        return images, questions, answers, mode_answers
    else:  # テストデータの場合
        images, questions = zip(*batch)
        
        images = torch.stack(images, 0)
        questions = list(questions)
        
        return images, questions

def main():
    # WandBの初期化
    wandb.init(project="vqa-project")
    wandb.config.update({
        "model": "VQAModel",
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "num_epochs": 50,
        "patience": 5,
    })

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データ拡張を含む画像の前処理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=train_transform)
    val_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=val_transform, answer=False)
    
    val_dataset.update_dict(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=custom_collate)

    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_train_acc = 0
    patience_counter = 0
    
    for epoch in range(wandb.config.num_epochs):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, device)
        val_loss, val_acc, val_simple_acc, val_time = eval(model, val_loader, device)
        
        scheduler.step(val_acc)
        
        # WandBにログを記録
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_simple_acc": train_simple_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_simple_acc": val_simple_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"【{epoch + 1}/{wandb.config.num_epochs}】")
        print(f"train time: {train_time:.2f}")
        print(f"train loss: {train_loss:.4f}")
        print(f"train acc: {train_acc:.4f}")
        print(f"train simple acc: {train_simple_acc:.4f}")
        
        # Early stopping
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= wandb.config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # # 最良のモデルを読み込む
    # model.load_state_dict(torch.load("best_model.pth"))
    
    # 検証データに対する最終予測（提出用）
    model.eval()
    submission = []
    with torch.no_grad():
        for image, question in tqdm(val_loader, desc="Generating predictions"):
            image = image.to(device)
            pred = model(image, question)
            pred = pred.argmax(1).cpu()
            submission.extend(pred.tolist())

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    np.save("submission.npy", submission)

    # WandBのartifactとしてモデルを保存
    wandb.save("best_model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()