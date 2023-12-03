import argparse
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class DenoisingScoreMatching(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x):
        score = self.model(x)
        return score

    def langevin_monte_carlo(self, N, K=100, alpha=0.1, device="cpu"):
        dim = self.input_dim
        noise_coeef = np.sqrt(2 * alpha)
        x = torch.randn(N, dim, device=device)
        noise = torch.randn(K, N, dim, device=device)
        for j in range(K):
            x = x + alpha * self.forward(x) + noise_coeef * noise[j]
            print(x)
        return x

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class LightningDenoisingScoreMatching(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.save_hyperparameters()
        self.model = DenoisingScoreMatching(input_dim, hidden_dim)
        self.sigma = sigma

    def one_step(self, batch, batch_idx):
        x = batch["x"]
        _ = batch_idx
        noise = self.sigma * torch.randn_like(x)
        x_hat = x + noise
        score = self.forward(x_hat)
        loss_fn = torch.nn.MSELoss(reduction="sum")
        loss = loss_fn(-1.0 * noise / (self.sigma * self.sigma), score)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log(
            "test_loss",
            loss,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

    def sample(self, N: int):
        return self.model.langevin_monte_carlo(N, device=self.device)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.from_numpy(x).float()
        return {"x": x}

    def __len__(self):
        return len(self.data)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 32):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        _ = stage
        data = np.loadtxt(self.dataset_path)
        self.dataset = MyDataset(data)
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        print(
            f"dataset size: total {len(self.dataset)}, train {train_size}, validation {val_size}, test {test_size}"
        )

    def train_dataloader(self):
        if len(self.train_dataset) != 0:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                drop_last=True,
            )
        else:
            raise Exception("length of dataset is zero.")

    def val_dataloader(self):
        if len(self.val_dataset) != 0:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )
        else:
            raise Exception("length of dataset is zero.")

    def test_dataloader(self):
        if len(self.test_dataset) != 0:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )
        else:
            raise Exception("length of dataset is zero.")


def command_train(args):
    data_module = MyDataModule(args.dataset_path)

    model = LightningDenoisingScoreMatching(args.input_dim, args.hidden_dim, args.sigma)

    logger = TensorBoardLogger(
        save_dir="data/lightning_logs",
        name="denoising_score_matching",
    )
    trainer = pl.Trainer(max_epochs=100, logger=logger)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def command_sample(args):
    model = LightningDenoisingScoreMatching.load_from_checkpoint(args.ckpt_path)
    model.eval()
    sampled_X = model.sample(N=10000)
    np.savetxt("data/dsm_sampled_X.far_x0.numpy.txt", sampled_X.cpu().detach().numpy())


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("--input_dim", type=int, default=2)
    parser_train.add_argument("--hidden_dim", type=int, default=128)
    parser_train.add_argument("--sigma", type=float, default=1.0)
    parser_train.add_argument(
        "--dataset_path", type=str, default="data/sampled_X.numpy.txt"
    )
    parser_train.set_defaults(handler=command_train)

    parser_sample = subparsers.add_parser("sample", help="see `sample -h`")
    parser_sample.add_argument("--ckpt_path", type=str, required=True, help="ckpt path")
    parser_sample.set_defaults(handler=command_sample)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
