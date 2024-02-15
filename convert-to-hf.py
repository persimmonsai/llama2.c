# convert from pytorch weights to huggingface safetensors

import sys
import os
import json
import torch
import safetensors.torch

if len(sys.argv) < 2:
	print(f"Usage: {sys.argv[0]} INFILE")
	sys.exit(1)

infile = sys.argv[1]
indir = os.path.dirname(infile)
outfile = sys.argv[2] if len(sys.argv) > 2 else indir + '/' + 'model.safetensors'

n_heads = None
if os.path.exists(indir + '/' + 'config.json'):
	with open(indir + '/' + 'config.json') as f:
		config = json.load(f)
		if 'num_attention_heads' in config:
			n_heads = config['num_attention_heads']
if n_heads is None and os.path.exists(indir + '/' + 'params.json'):
	with open(indir + '/' + 'params.json') as f:
		config = json.load(f)
		if 'n_heads' in config:
			n_heads = config['n_heads']

if n_heads is None:
	print("Number of attention heads unknown")
	sys.exit(0)

x = torch.load(sys.argv[1], map_location='cpu')
c = x['model']

# HuggingFace needs the weights permuted.
# See: https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
def hf_permute(w, n_heads, dim1, dim2):
	return w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

name_map = {
	"output": "lm_head", "tok_embeddings": "embed_tokens", "feed_forward": "mlp",
	"attention_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
	"attention.wq": "self_attn.q_proj", "attention.wk": "self_attn.k_proj", "attention.wv": "self_attn.v_proj", "attention.wo": "self_attn.o_proj",
	"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"
}

for k in list(c.keys()):
	newk = k
	for n in name_map.keys(): newk = newk.replace(n + '.', name_map[n] + '.')

	if False == newk.startswith('lm_head.'): newk = 'model.' + newk

	if newk.find('.q_proj.') >=0:
		dim = c[k].shape[0]
		c[newk] = hf_permute(c[k], n_heads, dim, dim)
	elif newk.find('.k_proj.') >=0:
		dim1 = c[k].shape[0]
		dim2 = c[k].shape[1]
		c[newk] = hf_permute(c[k], n_heads * dim1//dim2, dim1, dim2)
	else:
		c[newk] = c[k].clone().detach()
	del c[k]
safetensors.torch.save_file(c, outfile, metadata = {"format": "pt"})
