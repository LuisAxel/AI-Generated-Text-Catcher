import string
import torch
import transformers
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

class DistilBERTClass(torch.nn.Module):
    """
    Custom PyTorch model class using DistilBERT for binary classification.
    """
    def __init__(self):
        """
        Initializes the DistilBERT model, along with additional layers
        for classification and regularization.
        """
        super(DistilBERTClass, self).__init__()
        self.l1 = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for inputs.
            token_type_ids (torch.Tensor): Token type IDs for inputs.

        Returns:
            torch.Tensor: Probability scores after sigmoid activation.
        """
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output


def initialize_model(model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, device: torch.device) -> tuple:
    """
    Initializes the model and tokenizer from saved files.

    Args:
        model (torch.nn.Module): The model to initialize.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to initialize.
        device (torch.device): The device to use for the model (CPU/GPU).

    Returns:
        tuple: Initialized model and tokenizer, or (None, None) if an error occurs.
    """
    vocab_file = 'vocab_distilbert_writings.bin'
    model_file = 'pytorch_distilbert.bin'

    try:
        model, tokenizer = load_model_and_tokenizer(device, model_file, vocab_file)
        model.to(device)
        model.eval()
        print("Model and tokenizer initialized successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None


def load_model_and_tokenizer(device: torch.device, model_file: str, vocab_file: str) -> tuple:
    """
    Loads the model and tokenizer from the specified files.

    Args:
        device (torch.device): The device to map the model to.
        model_file (str): Path to the saved model file.
        vocab_file (str): Path to the vocabulary file for the tokenizer.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    tokenizer = transformers.BertTokenizer.from_pretrained(vocab_file, truncation=True, do_lower_case=True)
    model = DistilBERTClass()
    model.load_state_dict(torch.load(model_file, map_location=device))
    return model, tokenizer


def segment_text(text: str, segment_length: int = 20) -> list:
    """
    Segments a text into smaller chunks of a specified length.

    Args:
        text (str): The input text to segment.
        segment_length (int): The length of each segment (default: 20).

    Returns:
        list: A list of text segments.
    """
    words = text.split()
    segments = [(' '.join(words[i:i + segment_length]))
                for i in range(0, len(words), segment_length)]
    return segments


def test_essay(essay: str, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, max_len: int, device: torch.device) -> list:
    """
    Evaluates an essay using the model and tokenizer.

    Args:
        essay (str): The input essay text to evaluate.
        model (torch.nn.Module): The pre-trained model for predictions.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for input processing.
        max_len (int): The maximum token length for the tokenizer.
        device (torch.device): The device to use for computation (CPU/GPU).

    Returns:
        list: Predictions for each segmented chunk of the essay.
    """
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

        all_predictions.append(outputs)

    return all_predictions