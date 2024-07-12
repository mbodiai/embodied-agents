import json
from transformers import AutoProcessor

model_id = "google/paligemma-3b-mix-224"

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Assuming the processor includes a tokenizer
tokenizer = processor.tokenizer

# Create mappings
ra_to_token_map = {}
token_to_ra_map = {}
token_to_id_map = {}

INITIAL_OFFSET = 7

ra_id = 0

def should_include_token(token):
    # Include only tokens that start with '<'
    return token.startswith('<')

while len(ra_to_token_map) < 256:
    token_id = INITIAL_OFFSET + ra_id
    token = tokenizer.convert_ids_to_tokens(token_id)
    if should_include_token(token):
        ra_name = f"ra_{len(ra_to_token_map)}"
        ra_to_token_map[ra_name] = token
        token_to_ra_map[token] = ra_name
        token_to_id_map[token] = token_id
    ra_id += 1

# Save these mappings
with open('ra_to_token_map.json', 'w') as f:
    json.dump(ra_to_token_map, f)

with open('token_to_ra_map.json', 'w') as f:
    json.dump(token_to_ra_map, f)

# Save the tokenizer vocabulary
vocab = tokenizer.get_vocab()
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)

# Save the token to token_id map
with open('valid_tokens.json', 'w') as f:
    json.dump(token_to_id_map, f)

print("Mappings and vocabulary created and saved successfully.")
