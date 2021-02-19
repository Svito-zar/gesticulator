from bert_embedding import BertEmbedding
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
"""
The BERT implementation that was used for the paper is now deprecated:
    https://github.com/imgarylai/bert-embedding

Therefore the implementation has been updated to use HuggingFace's BERT implementation.

This script tests whether the HuggingFace BERT model produces the same word embeddings
as the original implementation that was used for the paper.
"""

sentence = "Are you a cool cat?"

# Old BERT model:
old_bert_model = BertEmbedding(max_seq_length = 100, 
    model = 'bert_12_768_12',
    dataset_name = 'book_corpus_wiki_en_cased')

input_to_bert, old_embeddings = old_bert_model([sentence])[0]
old_embeddings = torch.Tensor(old_embeddings)
print("Input to BERT:\t", input_to_bert)
print("------------------------------------------------")
print("Old embedding shape:\t\t", old_embeddings.shape)
print("------------------------------------------------")
# New BERT model:
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
new_model = BertModel.from_pretrained('bert-base-cased')

encoded_tokens = tokenizer.encode(sentence)
# Add batch dimension
encoded_tokens = torch.tensor(encoded_tokens).unsqueeze(0)
# The last hidden-state is the first element of the output tuple
new_embeddings = new_model(encoded_tokens)[0].squeeze()
print("New embedding shape:\t\t", new_embeddings.shape)

# The first and the last embedding vectors in `new_embedding` correspond to special tokens
# Remove the special tokens:
new_embeddings = new_embeddings[1:-1]
print("With special tokens removed:\t", new_embeddings.shape)
print("------------------------------------------------")
diff = torch.abs(new_embeddings - old_embeddings)
max_diff = torch.max(diff).item()
print("Max elementwise difference between two embeddings: ", max_diff)

print("------------------------------------------------")

print("Checking difference for mismatched input:")
sentence = "Are you a nice cat?"
input_to_bert, old_embeddings = old_bert_model([sentence])[0]
print("Input to BERT:\t", input_to_bert)
old_embeddings = torch.Tensor(old_embeddings)
diff = torch.abs(new_embeddings - old_embeddings)
max_diff = torch.max(diff).item()
print("Max elementwise difference between _mismatched_ embeddings: ", max_diff)
