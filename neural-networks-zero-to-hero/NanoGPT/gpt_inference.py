from gpt import BigramLanguageModel, decode, device
import torch

PATH = "your_path"
model = BigramLanguageModel()
model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
model.eval()
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
