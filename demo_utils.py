import string
import torch
import transformers

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output

def load_model_and_tokenizer(device, model_file, vocab_file):
    tokenizer = transformers.BertTokenizer.from_pretrained(vocab_file, truncation=True, do_lower_case=True)
    model = torch.load(model_file, map_location=device)
    return model, tokenizer

def segment_text(text: str, segment_length: int = 20) -> list:
    words = text.split()
    segments = [(' '.join(words[i:i + segment_length]))
                for i in range(0, len(words), segment_length)]
    return segments

def test_essay(essay, model, tokenizer, max_len, device):
    essay = essay.lower().translate(str.maketrans('', '', string.punctuation))
    chunks = segment_text(essay)

    model.eval()
    all_predictions = []

    for chunk in chunks:
        inputs = tokenizer.encode_plus(
            chunk,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids)
            outputs = outputs.squeeze()

        predicted_prob = outputs.round()
        all_predictions.append(predicted_prob)

    return all_predictions
