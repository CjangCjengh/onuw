import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config


class WerewolfBeliefModelConfig(PretrainedConfig):
    model_type = 'werewolf_belief'
    
    def __init__(
        self,
        num_players=5,
        d_model=512,
        n_head=8,
        n_layer=8,
        dropout=0.1,
        max_seq_length=256,
        tone_types=8,
        face_types=8,
        action_types=9,
        **kwargs
    ):
        self.num_players = num_players
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.tone_types = tone_types
        self.face_types = face_types
        self.action_types = action_types
        super().__init__(**kwargs)

class WerewolfBeliefModel(PreTrainedModel):
    config_class = WerewolfBeliefModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        gpt2_config = GPT2Config(
            n_embd=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_layer,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            layer_norm_epsilon=1e-5,
            use_cache=False,
        )

        self.subject_embedding = nn.Embedding(config.num_players, config.d_model)
        self.object_embedding = nn.Embedding(config.num_players, config.d_model)

        self.action_embedding = nn.Embedding(config.action_types, config.d_model)

        self.tone_embedding = nn.Embedding(config.tone_types, config.d_model)
        self.face_embedding = nn.Embedding(config.face_types, config.d_model)

        self.position_embeddings = nn.Embedding(config.max_seq_length, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # decoder-only
        self.transformer_layers = nn.ModuleList([
            GPT2Block(gpt2_config) for _ in range(config.n_layer)
        ])

        self.output_linear = nn.Linear(
            config.d_model, 
            config.num_players * config.num_players
        )

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self, subject_ids, action_ids, object_ids, tone_ids, face_ids):
        batch_size, seq_len = subject_ids.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device=subject_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        subject_embeddings = self.subject_embedding(subject_ids)
        object_embeddings = self.object_embedding(object_ids)
        action_embeddings = self.action_embedding(action_ids)
        tone_embeddings = self.tone_embedding(tone_ids)
        face_embeddings = self.face_embedding(face_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = subject_embeddings + action_embeddings + object_embeddings + \
                     tone_embeddings + face_embeddings + position_embeddings
        
        return self.dropout(self.layer_norm(embeddings))

    def forward(
        self,
        subject_ids,
        action_ids,
        object_ids,
        tone_ids,
        face_ids,
        attention_mask=None,
        labels=None
    ):
        batch_size, seq_len = subject_ids.shape

        hidden_states = self.get_input_embeddings(subject_ids, action_ids, object_ids, tone_ids, face_ids)

        for layer in self.transformer_layers:
            hidden_states, = layer(hidden_states)

        logits = self.output_linear(hidden_states)

        # [batch_size, seq_len, num_players, num_players]
        logits = logits.view(
            batch_size, seq_len, self.config.num_players, self.config.num_players
        )

        belief_matrix = F.softmax(logits, dim=-1)
        
        loss = None
        if labels is not None:
            pred_flat = belief_matrix.view(-1, self.config.num_players)
            labels_flat = labels.view(-1, self.config.num_players)
            
            # Compute loss only on non-masked positions
            if attention_mask is not None:
                # [batch_size, seq_len] -> [batch_size, seq_len, num_players]
                expanded_mask = attention_mask.unsqueeze(-1).expand(
                    -1, -1, belief_matrix.size(2)
                )

                active_loss = expanded_mask.reshape(-1) == 1
                active_logits = pred_flat[active_loss]
                active_labels = labels_flat[active_loss]
                loss = F.kl_div(active_logits.log(), active_labels, reduction='batchmean')
            else:
                loss = F.kl_div(pred_flat.log(), labels_flat, reduction='batchmean')

        return {'belief_matrix': belief_matrix, 'loss': loss} if labels is not None else {'belief_matrix': belief_matrix}
