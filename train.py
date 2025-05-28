import torch
from torch import nn, optim
import argparse

from utils import get_model, train_model, plot


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="finetune")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pretrain", dest="pretrain", action="store_true", help="使用预训练模型")
    p.add_argument("--no-pretrain", dest="pretrain", action="store_false", help="不使用预训练模型")
    p.set_defaults(pretrain=True)
    args = p.parse_args()

    model = get_model(pretrained=args.pretrain).to(device)

    params = [
        {'params': model.fc.parameters(), 'lr': args.lr},  
        {'params': [p for n, p in model.named_parameters() if "fc" not in n], 'lr': args.lr / 10 if args.pretrain else args.lr}  
    ]

    optimizer = optim.SGD(params, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = nn.CrossEntropyLoss()
    
    record = train_model(model, optimizer, scheduler, criterion, 
            exp_name=args.exp, epochs=args.epochs, batch_size=args.batch_size, device=device, save_best=True)

    plot(record, args.exp, {'batch_size': args.batch_size, 'lr': args.lr, 'step_size': args.step_size, 'gamma': args.gamma, 'weight_decay': args.weight_decay}, save_path='training_plots_finetune' if args.pretrain else 'training_plots_random')

if __name__ == '__main__':
    main()
