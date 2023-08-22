import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from utils import get_train_args, get_data, get_model

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

args = get_train_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs

device = torch.device("cuda" if args.gpu else "cpu")

dataloaders, image_datasets, _ = get_data(data_dir)

model = get_model(arch, hidden_units)

criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

model.to(device)

for epoch in range(epochs):
    train_loss = 0
    train_accuracy = 0
    valid_loss = 0
    valid_accuracy = 0

    model.train()
    for images, labels in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders["valid"]:
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            loss = criterion(log_ps, labels)

            valid_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    train_loss /= len(dataloaders["train"])
    train_accuracy /= len(dataloaders["train"])
    valid_loss /= len(dataloaders["valid"])
    valid_accuracy /= len(dataloaders["valid"])

    print(
        f"train_loss: {train_loss:.4f}, "
        f"train_accuracy: {train_accuracy:.4f}, "
        f"valid_loss: {valid_loss:.4f}, "
        f"valid_accuracy: {valid_accuracy:.4f}"
    )


torch.save(
    {
        "arch": arch,
        "hidden_units": hidden_units,
        "state_dict": model.state_dict(),
        "class_to_idx": image_datasets["train"].class_to_idx,
    },
    save_dir + "/checkpoint.pth",
)

print("Training complete!")
