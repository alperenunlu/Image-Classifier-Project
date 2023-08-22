import torch

from PIL import Image

import json

from utils import get_predict_args, get_predict_model, process_image, load_checkpoint

torch.manual_seed(0)

args = get_predict_args()

image_path = args.image_path
checkpoint_path = args.checkpoint
top_k = args.top_k
category_names = args.category_names

device = torch.device("cuda" if args.gpu else "cpu")

if args.previous_model:
    model = load_checkpoint(checkpoint_path, device)
else:
    model = get_predict_model(checkpoint_path)
model.to(device)

class_to_idx = model.class_to_idx

with open(category_names, "r") as f:
    cat_to_name = json.load(f)

image = Image.open(image_path)
image = process_image(image)
image = torch.from_numpy(image).float()
image = image.unsqueeze(0)

image = image.to(device)

model.eval()
with torch.no_grad():
    log_ps = model(image)

ps = torch.exp(log_ps)
top_p, top_class = ps.topk(top_k, dim=1)

top_p = top_p.cpu().numpy().squeeze()
top_class = top_class.cpu().numpy().squeeze()

idx_to_class = {v: k for k, v in class_to_idx.items()}

for i in range(top_k):
    print(f"{i+1}: {cat_to_name[idx_to_class[top_class[i]]]} ({top_p[i] * 100:.2f}%)")
