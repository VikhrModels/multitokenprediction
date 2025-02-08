import torch
from transformers import LlamaForCausalLM

def generate_multitoken(model, input_ids, max_new_tokens, n_future_tokens):
    """
    Генерирует текст, используя модель с патченной forward, которая возвращает логиты
    формы (batch_size, seq_len, n_future_tokens, vocab_size).
    На каждом шаге выбираются предсказания для всех future-токенов, которые затем дописываются к последовательности.
    """
    generated = input_ids  # Здесь input_ids уже является тензором!
    steps = max_new_tokens // n_future_tokens
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            outputs = model(generated)
            # last_logits имеет форму: (batch_size, n_future_tokens, vocab_size)
            last_logits = outputs[0, -1, :, :]
            # Выбираем argmax для каждого из n_future_tokens
            predicted_tokens = torch.argmax(last_logits, dim=-1)  # (batch_size, n_future_tokens)
            # Если батч равен 1, убеждаемся, что predicted_tokens имеет форму (1, n_future_tokens)
            if predicted_tokens.dim() == 1:
                predicted_tokens = predicted_tokens.unsqueeze(0)
            generated = torch.cat([generated, predicted_tokens], dim=1)
    return generated



def patch_llama_for_multitoken():
    import copy
    def patched_forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]  # ожидаем форму (batch_size, seq_len, hidden_size)
        n_future_tokens = getattr(self.config, "n_future_tokens", 1)

        if n_future_tokens > 1:
            trunk_states = hidden_states
            latents = [trunk_states]

            if not hasattr(self, "extra_heads"):
                last_layer = self.model.layers[-1]
                self.extra_heads = torch.nn.ModuleList([
                    copy.deepcopy(last_layer) for _ in range(n_future_tokens - 1)
                ])

            if "position_ids" not in kwargs or kwargs["position_ids"] is None:
                batch_size, seq_len, _ = trunk_states.shape
                kwargs["position_ids"] = torch.arange(seq_len, device=trunk_states.device)\
                    .unsqueeze(0).expand(batch_size, seq_len)

            for head in self.extra_heads:
                if isinstance(trunk_states, tuple):
                    trunk_states = trunk_states[0]
                batch_size, seq_len, _ = trunk_states.shape
                local_kwargs = kwargs.copy()
                local_kwargs['position_ids'] = torch.arange(seq_len, device=trunk_states.device)\
                    .unsqueeze(0).expand(batch_size, seq_len)
                if attention_mask is not None:
                    if attention_mask.ndim == 2:
                        local_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    else:
                        local_attention_mask = attention_mask
                    local_attention_mask = local_attention_mask.float()
                else:
                    local_attention_mask = None
                output = head(trunk_states, attention_mask=local_attention_mask, **local_kwargs)
                if isinstance(output, tuple):
                    trunk_states = output[0]
                else:
                    trunk_states = output
                latents.append(trunk_states)

            hidden_states = torch.stack(latents, dim=2)  # (batch_size, seq_len, n_future_tokens, hidden_size)
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return logits

    LlamaForCausalLM.forward = patched_forward
    print("LlamaForCausalLM patched for multi-token generation.")
