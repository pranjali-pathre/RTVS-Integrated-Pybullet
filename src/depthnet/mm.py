import numpy as np
import torch
import torch.nn.functional as F
from .losses import depth_loss
import pytorch_lightning as pl


class PLModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optim_kwargs, sched_kwargs):
        super().__init__()
        self.model = model
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = sched_kwargs
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, mode: str = "train"):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y, y_hat)
        self.log_dict(
            {
                f"{mode}_loss": loss.detach(),
                f"{mode}_l1loss": F.l1_loss(y_hat, y).detach(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, mode="test")

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, **self.scheduler_kwargs
        )
        return {
            "optimizer": self.optim,
            "lr_scheduler": {"scheduler": self.sched, "monitor": "val_loss"},
        }


class ModelManager:
    def __init__(self, config: dict, model: torch.nn.Module, dataset):
        self.config = config
        self.dataset = dataset

        self.train_dl, self.val_dl, self.test_dl = self.dataset.get_dataloaders(
            self.config["batch_size"], self.config["val_test_split"]
        )

        self.pl_module = PLModule(
            model, depth_loss, self.config["optim"], self.config["lr_scheduler"]
        )
        self._setup_pl_trainer()

    def train(self, ckpt_path: str = None):
        self.trainer.fit(
            self.pl_module, self.train_dl, self.val_dl, ckpt_path=ckpt_path
        )

    def test(self, ckpt_path: str = None):
        self.trainer.test(self.pl_module, self.test_dl, ckpt_path=ckpt_path)

    @torch.inference_mode()
    def predict(self, x, single=False, as_tensor=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.pl_module.device)
        if single:
            x = x.unsqueeze(0)
        out = self.pl_module(x).detach().cpu()
        if single:
            out = out.squeeze(0)
        if as_tensor:
            return out

        return out.numpy()

    def __call__(self, x, single=False, as_tensor=False):
        return self.predict(x, single=single, as_tensor=as_tensor)

    def _setup_pl_trainer(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.config["train_logs_dir"] + "/ckpts",
            filename="checkpoint_{epoch:03d}_{val_loss:08.4f}",
            save_top_k=50,
            monitor="val_loss",
            every_n_epochs=1,
            save_last=True,
            save_on_train_epoch_end=False,
        )

        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=self.config["gpus"],
            strategy="dp",
            callbacks=[checkpoint_callback, lr_monitor_callback],
            default_root_dir=self.config["train_logs_dir"],
            max_epochs=self.config["n_epochs"],
            benchmark=True,
            precision=32,
            log_every_n_steps=1,
            # fast_dev_run=True,
        )

    def load_weights(self, ckpt_path: str, strict=True):
        self.pl_module.load_state_dict(
            torch.load(ckpt_path, map_location=self.pl_module.device)["state_dict"],
            strict,
        )
