from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
import fsspec
from config import TrainerConfig, Snapshot

class Trainer:
    def __init__(self, trainer_config:TrainerConfig, 
                model, 
                optimizer, 
                local_rank: int, 
                global_rank: int, 
                train_loader: DataLoader, 
                test_loader: DataLoader = None
                ) -> None:
        self.config = trainer_config
        self.local_rank = local_rank
        self.global_rank = global_rank

        self.trainloader = train_loader
        self.testloader = test_loader

        self.epoch_run = 0
        self.model = model
        self.optimizer = optimizer
        self.save_every = self.config.save_every
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location='cpu')
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch.")
            return
        snapshot = Snapshot(**snapshot_data)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(snapshot.model_state)
        else:
            self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epoch_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epoch_run}")

    def _save_snapshot(self, epoch):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(model_state=raw_model.state_dict(), 
                            optimizer_state=self.optimizer.state_dict(), 
                            finished_epoch=epoch)
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)
        print("Saving snapshot at epoch:", epoch)

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)):
            _, loss = self.model(source, targets)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self, epoch: int, loader: DataLoader, train: bool = True):
        loader.sampler.set_epoch(epoch)
        for iter, (source, targets) in enumerate(loader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source=source, targets=targets, train=train)
            if iter % 100 == 0:
                print(f"GPU{self.local_rank} Epoch:{epoch} Iter:{iter} {step_type} Loss:{batch_loss}")
    
    def train(self, ):
        for epoch in range(self.epoch_run, self.config.max_epochs):
            epoch += 1
            self._run_epoch(epoch, self.trainloader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            if self.testloader:
                self._run_epoch(epoch, self.testloader, train=False)
