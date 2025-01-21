import torch
from transformers import AutoTokenizer, BertForMaskedLM

tok = AutoTokenizer.from_pretrained("bert-base-cased")
bert = BertForMaskedLM.from_pretrained("bert-base-cased")

input_idx = tok.encode(f"The {tok.mask_token} were the best rock band ever.")
logits = bert(torch.tensor([input_idx]))[0]
prediction = logits[0].argmax(dim=1)
print(tok.convert_ids_to_tokens(prediction[2].numpy().tolist()))