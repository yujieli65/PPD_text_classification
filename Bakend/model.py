# model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class BertWithTFIDF(nn.Module):
    def __init__(self, model_name, tfidf_dim, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size + tfidf_dim, num_labels)

    def forward(self, input_ids, attention_mask, tfidf_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        fused = torch.cat((cls_output, tfidf_features), dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)
