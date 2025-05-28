import torch
from torch import nn
import argparse

from utils import get_model, evaluate, get_dataloaders


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="finetune")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(f'{args.exp}_best.pth', map_location=device))
    model.to(device)

    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()

    loss, accuracy = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()