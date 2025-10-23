"""capstone_project.ipynb
import pandas as pd
df=pd.read_csv('/content/capstone_csv')
df

"""**INSTALL TRANSFORMERS & USE ROBERT**"""

!pip install transformers
from transformers import RobertaTokenizer
tokenizer=RobertaTokenizer.from_pretrained('roberta-base')

tokens=tokenizer(
    list(df['text']),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
print(tokens.keys())

import torch
labels = torch.tensor(df['label'].values)

from torch.utils.data import TensorDataset
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)

"""**TRAIN/TEST SPLIT**"""

import numpy as np
print('CUDA available:', torch.cuda.is_available())

!pip -q install datasets evaluate scikit-learn
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

idx = np.arange(len(df))
train_idx, val_idx = train_test_split(idx, test_size=0.2,
                                      stratify=df['label'], random_state=42)

def build_ds(indices):
    return Dataset.from_dict({
        'input_ids': tokens['input_ids'][indices].numpy(),
        'attention_mask': tokens['attention_mask'][indices].numpy(),
        'labels': labels[indices].numpy()
    })

ds = DatasetDict({
    'train': build_ds(train_idx),
    'val': build_ds(val_idx)
})

ds

from transformers import RobertaForSequenceClassification, set_seed
set_seed(42)

model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2
)

"""**LOAD ROBERT MODEL**"""

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def compute_metrics(eval_pred):
    logits, labels_np = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc  = accuracy_score(labels_np, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_np, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir="ai_vs_human_roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

"""**Evaluation**"""

metrics = trainer.evaluate()
print(metrics)

from sklearn.metrics import classification_report
predictions = trainer.predict(ds['val'])
pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids
print(classification_report(ds['val']['labels'], pred_labels))

"""**Confusion matrix**"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Human", "AI"],
            yticklabels=["Human", "AI"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

"""**ROC CURVE**"""

from sklearn.metrics import roc_curve, auc

probs = predictions.predictions[:, 1]
fpr, tpr, _ = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

"""**LIME EXPLAIN**"""

!pip install lime

import numpy as np
from lime.lime_text import LimeTextExplainer
import torch

class_names = ["Human-Written", "AI-Generated"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def lime_predict(texts):
    model.eval()
    results = []
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            results.append(probs)
    return np.array(results)

explainer = LimeTextExplainer(class_names=class_names)

sample_text = "This research paper is a thorough exploration of deep learning techniques."
exp = explainer.explain_instance(
    sample_text,
    lime_predict,
    num_features=10
)

exp.show_in_notebook(text=sample_text)

"""**GRADIO INTERFACE**"""

import gradio as gr

def predict_with_explanation(text):
    probs = lime_predict([text])[0]
    label_map = {1: "Human-Written", 0: "AI-Generated"}
    pred_label = label_map[int(probs.argmax())]
    exp = explainer.explain_instance(text, lime_predict, num_features=10)
    explanation_html = exp.as_html()
    return pred_label, float(probs.max()), explanation_html

gr.Interface(
    fn=predict_with_explanation,
    inputs=gr.Textbox(lines=5, placeholder="Paste your text here..."),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.HTML(label="Word Importance")
    ],
    title="AI vs Human Text Detector with RoBERTa + LIME",
    description="Paste text to see the prediction and which words influenced the decision."
).launch()

model.save_pretrained('final_roberta_model')
tokenizer.save_pretrained('final_roberta_model')

torch.save(model.state_dict(),'ai_vs_human_weights.pth')

from google.colab import files
files.download('ai_vs_human_weights.pth')
