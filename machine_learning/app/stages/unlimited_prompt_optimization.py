# unlimited_prompt_optimization.py

import re
import torch
from diffusers import StableDiffusionXLPipeline

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    re_attention = re.compile(
        r"""
        \\\(|
        \\\)|
        \\\[|
        \\]|
        \\\\|
        \\|
        \(|
        \[|
        :([+-]?[.\d]+)\)|
        \)|
        ]|
        [^\\()\[\]:]+|
        :
        """,
        re.X,
    )

    re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

    for m in re_attention.finditer(text):
        token = m.group(0)
        weight = m.group(1)

        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            # Apply custom weight from (text:weight)
            start = round_brackets.pop()
            try:
                w = float(weight)
            except ValueError:
                w = 1.0
            for i in range(start, len(res)):
                res[i][1] *= w
        elif token == ")" and len(round_brackets) > 0:
            # Apply default round bracket multiplier
            start = round_brackets.pop()
            for i in range(start, len(res)):
                res[i][1] *= round_bracket_multiplier
        elif token == "]" and len(square_brackets) > 0:
            # Apply default square bracket multiplier
            start = square_brackets.pop()
            for i in range(start, len(res)):
                res[i][1] *= square_bracket_multiplier
        else:
            parts = re.split(re_break, token)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    # Handle unmatched brackets
    for pos in round_brackets:
        for i in range(pos, len(res)):
            res[i][1] *= round_bracket_multiplier

    for pos in square_brackets:
        for i in range(pos, len(res)):
            res[i][1] *= square_bracket_multiplier

    if len(res) == 0:
        res = [["", 1.0]]

    # Merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

def get_prompts_tokens_with_weights(
    tokenizer,
    prompt: str = None
):
    """
    Get prompt token ids and weights; this function works for both prompt and negative prompt.
    """
    if (prompt is None) or (len(prompt) < 1):
        prompt = " "

    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []
    for word, weight in texts_and_weights:
        # Tokenize and discard the starting and ending tokens
        tokens = tokenizer(
            word,
            truncation=False,  # So that it tokenizes whatever length prompt
            add_special_tokens=False
        ).input_ids

        # Merge the new tokens into the all tokens holder: text_tokens
        text_tokens.extend(tokens)

        # Each token chunk will come with one weight, like ['red cat', 2.0]
        # Need to expand weight for each token.
        chunk_weights = [weight] * len(tokens)

        # Append the weights back to the weight holder: text_weights
        text_weights.extend(chunk_weights)
    return text_tokens, text_weights

def group_tokens_and_weights(
    token_ids: list,
    weights: list,
    bos_token_id: int,
    eos_token_id: int,
    pad_last_block=True
):
    """
    Produce tokens and weights in groups and pad the missing tokens.
    """
    max_length = 77  # SDXL uses a max length of 77 tokens (including BOS and EOS)
    chunk_size = max_length - 2  # Subtract BOS and EOS

    # This will be a 2D list
    new_token_ids = []
    new_weights = []
    while len(token_ids) >= chunk_size:
        # Get the first chunk_size tokens
        head_tokens = [token_ids.pop(0) for _ in range(chunk_size)]
        head_weights = [weights.pop(0) for _ in range(chunk_size)]

        # Extract token ids and weights
        temp_token_ids = [bos_token_id] + head_tokens + [eos_token_id]
        temp_weights = [1.0] + head_weights + [1.0]

        # Add token and weights chunk to the holder list
        new_token_ids.append(temp_token_ids)
        new_weights.append(temp_weights)

    # Padding the leftover tokens
    if len(token_ids) > 0:
        temp_token_ids = [bos_token_id] + token_ids + [eos_token_id]
        temp_weights = [1.0] + weights + [1.0]
        if pad_last_block:
            # Pad to full length
            padding_length = chunk_size - len(token_ids)
            temp_token_ids += [eos_token_id] * padding_length
            temp_weights += [1.0] * padding_length
        new_token_ids.append(temp_token_ids)
        new_weights.append(temp_weights)

    return new_token_ids, new_weights

def get_weighted_text_embeddings_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str = "",
    neg_prompt: str = "",
    pad_last_block=True
):
    """
    This function processes long prompts with weights, with no length limitation for Stable Diffusion XL.
    """
    # Tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )

    # Tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )
    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )

    # Ensure both token lists are the same length
    max_length = max(len(prompt_tokens), len(neg_prompt_tokens))
    max_length_2 = max(len(prompt_tokens_2), len(neg_prompt_tokens_2))

    # Pad token lists to the same length
    prompt_tokens += [pipe.tokenizer.eos_token_id] * (max_length - len(prompt_tokens))
    prompt_weights += [1.0] * (max_length - len(prompt_weights))
    neg_prompt_tokens += [pipe.tokenizer.eos_token_id] * (max_length - len(neg_prompt_tokens))
    neg_prompt_weights += [1.0] * (max_length - len(neg_prompt_weights))

    prompt_tokens_2 += [pipe.tokenizer_2.eos_token_id] * (max_length_2 - len(prompt_tokens_2))
    prompt_weights_2 += [1.0] * (max_length_2 - len(prompt_weights_2))
    neg_prompt_tokens_2 += [pipe.tokenizer_2.eos_token_id] * (max_length_2 - len(neg_prompt_tokens_2))
    neg_prompt_weights_2 += [1.0] * (max_length_2 - len(neg_prompt_weights_2))

    # Group tokens and weights
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy(),
        prompt_weights.copy(),
        pipe.tokenizer.bos_token_id,
        pipe.tokenizer.eos_token_id,
        pad_last_block=pad_last_block
    )
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy(),
        neg_prompt_weights.copy(),
        pipe.tokenizer.bos_token_id,
        pipe.tokenizer.eos_token_id,
        pad_last_block=pad_last_block
    )

    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy(),
        prompt_weights_2.copy(),
        pipe.tokenizer_2.bos_token_id,
        pipe.tokenizer_2.eos_token_id,
        pad_last_block=pad_last_block
    )
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy(),
        neg_prompt_weights_2.copy(),
        pipe.tokenizer_2.bos_token_id,
        pipe.tokenizer_2.eos_token_id,
        pad_last_block=pad_last_block
    )

    embeds = []
    neg_embeds = []
    pooled_prompt_embeds_list = []
    negative_pooled_prompt_embeds_list = []

    # Process prompt embeddings
    for i in range(len(prompt_token_groups)):
        # Get token tensors
        token_tensor = torch.tensor(
            [prompt_token_groups[i]], dtype=torch.long, device=pipe.device
        )
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]], dtype=torch.long, device=pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i], dtype=torch.float32, device=pipe.device
        )
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i], dtype=torch.float32, device=pipe.device
        )

        # Encode with first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor,
            output_hidden_states=True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]

        # Encode with second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2,
            output_hidden_states=True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds_list.append(prompt_embeds_2[0])

        # Squeeze the batch dimension
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.squeeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.squeeze(0)

        # Apply weights to embeddings
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                prompt_embeds_1_hidden_states[j] = (
                    prompt_embeds_1_hidden_states[-1] + (prompt_embeds_1_hidden_states[j] - prompt_embeds_1_hidden_states[-1]) * weight_tensor[j]
                )
            if weight_tensor_2[j] != 1.0:
                prompt_embeds_2_hidden_states[j] = (
                    prompt_embeds_2_hidden_states[-1] + (prompt_embeds_2_hidden_states[j] - prompt_embeds_2_hidden_states[-1]) * weight_tensor_2[j]
                )

        # Combine embeddings
        token_embedding = torch.cat([prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states], dim=-1)

        embeds.append(token_embedding.unsqueeze(0))

    # Average pooled embeddings
    pooled_prompt_embeds = torch.mean(torch.stack(pooled_prompt_embeds_list, dim=0), dim=0)

    # Process negative prompt embeddings
    for i in range(len(neg_prompt_token_groups)):
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]], dtype=torch.long, device=pipe.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]], dtype=torch.long, device=pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i], dtype=torch.float32, device=pipe.device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i], dtype=torch.float32, device=pipe.device
        )

        # Encode with first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor,
            output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

        # Encode with second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2,
            output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds_list.append(neg_prompt_embeds_2[0])

        # Squeeze the batch dimension
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.squeeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.squeeze(0)

        # Apply weights to embeddings
        for j in range(len(neg_weight_tensor)):
            if neg_weight_tensor[j] != 1.0:
                neg_prompt_embeds_1_hidden_states[j] = (
                    neg_prompt_embeds_1_hidden_states[-1] + (neg_prompt_embeds_1_hidden_states[j] - neg_prompt_embeds_1_hidden_states[-1]) * neg_weight_tensor[j]
                )
            if neg_weight_tensor_2[j] != 1.0:
                neg_prompt_embeds_2_hidden_states[j] = (
                    neg_prompt_embeds_2_hidden_states[-1] + (neg_prompt_embeds_2_hidden_states[j] - neg_prompt_embeds_2_hidden_states[-1]) * neg_weight_tensor_2[j]
                )

        # Combine embeddings
        neg_token_embedding = torch.cat([neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states], dim=-1)

        neg_embeds.append(neg_token_embedding.unsqueeze(0))

    # Average negative pooled embeddings
    negative_pooled_prompt_embeds = torch.mean(torch.stack(negative_pooled_prompt_embeds_list, dim=0), dim=0)

    # Concatenate embeddings
    prompt_embeds = torch.cat(embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
