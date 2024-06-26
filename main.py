import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from model import GPT
from trainer import Trainer
from char_dataset import CharDataset
from config import DataConfig, TrainerConfig, OptimizerConfig, GPTConfig
import yaml

def create_optimizer(model: torch.nn.Module, opt_config: OptimizerConfig):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    # 对权重参数分组，LN，Emebdding，偏置和位置编码不weight decay
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('in_proj_weight'):
                # MHA projection layer
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('pos_embed'):
                # positional embedding shouldn't be decayed
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": opt_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95))
    return optimizer

def main():
    with open("./gpt2_train_cfg.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # 设置DDP环境
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # 读取配置
    gpt_cfg: GPTConfig = GPTConfig(**cfg['gpt_config'])
    opt_cfg: OptimizerConfig = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg: DataConfig = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])
    # 创建数据集和加载器
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    trainset, testset = random_split(dataset, [train_len, len(dataset) - train_len])
    trainloader = DataLoader(trainset, 
                            batch_size=data_cfg.batch_size, 
                            pin_memory=True, 
                            shuffle=False, 
                            num_workers=data_cfg.data_loader_workers, 
                            sampler=DistributedSampler(trainset)
                            )
    testloader = DataLoader(
                            testset,
                            batch_size=data_cfg.batch_size,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=data_cfg.data_loader_workers,
                            sampler=DistributedSampler(testset)
                            )
    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)

    # 模型分配到不同GPU上
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])    
    trainer = Trainer(trainer_cfg, model, optimizer, global_rank=global_rank, local_rank=local_rank,
    train_loader=trainloader, test_loader=testloader)
    trainer.train()

    destroy_process_group()

if __name__ == "__main__":
    main()
