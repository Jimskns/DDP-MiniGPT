## DDP-MiniGPT
Derived from [Link](https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp/mingpt)

## 执行命令
**单机多卡**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nodes=1 main.py
```
**多机多卡**
```bash
torchrun --nproc_per_node=4 --nnodes=3 --node_rank=0 --master_addr=192.168.0.* --master_port=12345 main.py
```
