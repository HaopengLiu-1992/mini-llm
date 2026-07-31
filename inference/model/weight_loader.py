import torch
from transformers import GPT2LMHeadModel

def load_hf_gpt2_weights(custom_model, hf_model):
    hf_transformer = hf_model.transformer

    _check_config(custom_model, hf_model.config)

    with torch.no_grad():
        custom_model.wte.weight.copy_(
            hf_transformer.wte.weight
        )
        custom_model.wpe.weight.copy_(
            hf_transformer.wpe.weight
        )

        for custom_layer, hf_layer in zip(
            custom_model.transformer_layers,
            hf_transformer.h,
        ):
            custom_layer.layer_norm1.weight.copy_(
                hf_layer.ln_1.weight
            )
            custom_layer.layer_norm1.bias.copy_(
                hf_layer.ln_1.bias
            )

            q_weight, k_weight, v_weight = hf_layer.attn.c_attn.weight.split(
                custom_model.dim,
                dim=1,
            )
            q_bias, k_bias, v_bias = hf_layer.attn.c_attn.bias.split(
                custom_model.dim,
                dim=0,
            )

            custom_layer.mha.wq.weight.copy_(q_weight.T)
            custom_layer.mha.wk.weight.copy_(k_weight.T)
            custom_layer.mha.wv.weight.copy_(v_weight.T)
            custom_layer.mha.wq.bias.copy_(q_bias)
            custom_layer.mha.wk.bias.copy_(k_bias)
            custom_layer.mha.wv.bias.copy_(v_bias)

            custom_layer.mha.wo.weight.copy_(
                hf_layer.attn.c_proj.weight.T
            )
            custom_layer.mha.wo.bias.copy_(
                hf_layer.attn.c_proj.bias
            )

            custom_layer.layer_norm2.weight.copy_(
                hf_layer.ln_2.weight
            )
            custom_layer.layer_norm2.bias.copy_(
                hf_layer.ln_2.bias
            )

            custom_layer.ffn.w1.weight.copy_(
                hf_layer.mlp.c_fc.weight.T
            )
            custom_layer.ffn.w1.bias.copy_(
                hf_layer.mlp.c_fc.bias
            )
            custom_layer.ffn.w2.weight.copy_(
                hf_layer.mlp.c_proj.weight.T
            )
            custom_layer.ffn.w2.bias.copy_(
                hf_layer.mlp.c_proj.bias
            )

        custom_model.layer_norm.weight.copy_(
            hf_transformer.ln_f.weight
        )
        custom_model.layer_norm.bias.copy_(
            hf_transformer.ln_f.bias
        )

        custom_model.lm.weight.copy_(
            hf_model.lm_head.weight
        )
        if custom_model.lm.bias is not None:
            custom_model.lm.bias.zero_()

    custom_model.eval()
    return custom_model


def load_pretrained_gpt2(custom_model, model_name):
    

    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    return load_hf_gpt2_weights(custom_model, hf_model)


def _check_config(custom_model, hf_config):
    expected = {
        "vocab_size": hf_config.vocab_size,
        "max_seq_len": hf_config.n_positions,
        "dim": hf_config.n_embd,
        "n_heads": hf_config.n_head,
        "n_layers": hf_config.n_layer,
    }

    actual = {
        "vocab_size": custom_model.vocab_size,
        "max_seq_len": custom_model.max_seq_len,
        "dim": custom_model.dim,
        "n_heads": custom_model.n_heads,
        "n_layers": custom_model.n_layers,
    }

    mismatches = {
        name: (actual[name], expected[name])
        for name in expected
        if actual[name] != expected[name]
    }

    if mismatches:
        raise ValueError(
            "Custom GPT2 config does not match HuggingFace config: "
            f"{mismatches}"
        )

