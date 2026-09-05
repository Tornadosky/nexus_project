"""Offline model-loading/inference check, not an experimental generation."""
import json, sys, time
from pathlib import Path
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
lock = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
if out.exists():
    raise FileExistsError(out)
torch.set_num_threads(4)
t0 = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(lock['local_path'], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(lock['local_path'],
    local_files_only=True, torch_dtype=torch.float32, device_map='cpu').eval()
inputs = tokenizer('Return the word ready.', return_tensors='pt')
with torch.inference_mode():
    result = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                            pad_token_id=tokenizer.eos_token_id)
record = dict(pass_inference=True, model=lock['model'], revision=lock['revision'],
    device='cpu', dtype='float32', torch=torch.__version__,
    transformers=transformers.__version__, seconds=time.monotonic()-t0,
    generated_tokens=int(result.shape[1]-inputs.input_ids.shape[1]),
    scientific_sample=False)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
