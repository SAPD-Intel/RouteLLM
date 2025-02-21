import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim,
        num_classes,
        use_proj,
        use_openai_embeddings=False,  # Default: Hugging Face embeddings
        embedding_model_name="intfloat/e5-base-v2",  # Match notebook
        hf_token=None,  # Hugging Face API token
    ):
        super().__init__()
        self.use_proj = use_proj
        self.use_openai_embeddings = use_openai_embeddings
        self.hf_token = hf_token
        self.embedding_model_name = embedding_model_name

        # Model embedding matrix
        self.P = torch.nn.Embedding(num_models, dim)

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert text_dim == dim, f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Linear(dim, num_classes, bias=False)

        if not self.use_openai_embeddings:
            logger.info(f"Loading Hugging Face tokenizer and model: {self.embedding_model_name}")

            # Load tokenizer & model exactly as in the notebook
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                token=hf_token  # Use `token` instead of `use_auth_token`
            )
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                token=hf_token  # Use `token` instead of `use_auth_token`
            )
            self.embedding_model.eval()  # Set to inference mode
            self.embedding_model.to(self.get_device())

    def get_device(self):
        return self.P.weight.device

    def get_prompt_embedding(self, prompt):
        """Generate sentence embedding using mean pooling (matches notebook)."""
        logger.info(f"Generating embedding for prompt: {prompt[:30]}...")
        
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.get_device())

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        # Mean pooling over token embeddings
        prompt_embed = last_hidden_state.mean(dim=1).squeeze()
        
        return prompt_embed

    def forward(self, model_id, prompt):
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())
        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)
        prompt_embed = self.get_prompt_embedding(prompt)

        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        raw_diff = logits[0] - logits[1]
        winrate = torch.sigmoid(raw_diff).item()
        logger.info(
            f"For prompt: '{prompt[:30]}...', logits: {[float(x) for x in logits]}, "
            f"raw difference: {raw_diff:.4f}, winrate (sigmoid): {winrate:.4f}"
        )
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
