import itertools
from tqdm import tqdm
import torch
from torch import nn, optim
import json
import argparse

from utils import get_model, train_search, plot

class GridSearcher:
    def __init__(self, opts, defaults):
        self.combinations = self.generate_combinations(opts, defaults)
        self.results = []

    @staticmethod
    def generate_combinations(hyper_param_opts, hyper_param_defaults):
        for key in hyper_param_opts.keys():
            if len(hyper_param_opts[key]) == 0:
                hyper_param_opts.pop(key)
        for key in hyper_param_defaults.keys():
            if key not in hyper_param_opts.keys() or len(hyper_param_opts[key]) == 0:
                hyper_param_opts[key] = [
                    hyper_param_defaults[key]] 

        combinations = []
        for values in itertools.product(*hyper_param_opts.values()):
            combination = dict(zip(hyper_param_opts.keys(), values))
            combinations.append(combination)
        return combinations
    
    def search(self, exp_name, epochs, pretrained=True):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for combination in tqdm(self.combinations):
            model = get_model(pretrained=pretrained).to(device)
            params = [
                {'params': model.fc.parameters(), 'lr': combination['lr']},  
                {'params': [p for n, p in model.named_parameters() if "fc" not in n], 'lr': combination['lr'] / 10 if pretrained else combination['lr']}  
            ]
            optimizer = optim.SGD(params, momentum=0.9, weight_decay=combination['weight_decay'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=combination['step_size'], gamma=combination['gamma'])
            criterion = nn.CrossEntropyLoss()
            record = train_search(model, optimizer, scheduler, criterion, epochs=epochs, batch_size=combination['batch_size'], device=device)
            plot(record, exp_name, {'batch_size': combination['batch_size'], 'lr': combination['lr'], 'step_size': combination['step_size'], 'gamma': combination['gamma'], 'weight_decay': combination['weight_decay']},
                  save_path='training_plots_finetune' if pretrained else 'training_plots_random')
            self.results.append((combination, record['best_val_acc']))

        self.results.sort(key=lambda x: x[1], reverse=True)
        return self.results
    
hyper_param_defaults = {
    "batch_size": 128,
    "lr": 0.01,
    "step_size": 10,
    "gamma": 0.5,
    "weight_decay": 1e-4,
} 

hyper_param_opts = {
    "batch_size": [256, 128, 64],
    "lr": [0.01, 0.001],
    "step_size": [10, 30],
    "gamma": [0.5, 0.1],
    "weight_decay": [1e-4, 1e-5, 0.0],
}

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="finetune")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--pretrain", dest="pretrain", action="store_true", help="使用预训练模型")
    p.add_argument("--no-pretrain", dest="pretrain", action="store_false", help="不使用预训练模型")
    p.set_defaults(pretrain=True)
    args = p.parse_args()

    searcher = GridSearcher(hyper_param_opts, hyper_param_defaults)
    results = searcher.search(args.exp, epochs=args.epochs, pretrained=args.pretrain)
    with open(f"gridsearch_results_{args.exp}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
