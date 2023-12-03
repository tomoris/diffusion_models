import argparse
from typing import Any

import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class DiffusionModel(torch.nn.Module):
    def __init__(self, input_dim, t_num, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.t_num = t_num
        self.hidden_dim = hidden_dim

        self.t_embed = torch.nn.Embedding(self.t_num, self.hidden_dim)

        self.x_embed = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x, t):
        x_embed = self.x_embed(x)

        t_embed = self.t_embed(t)

        predicted_noise = self.model(torch.cat([x_embed, t_embed], dim=1))
        return predicted_noise

    def sample(self, ts, alpha, beta, beta_bar, N, device="cpu"):
        assert ts.dim() == 1
        with torch.no_grad():
            sorted_ts = ts.sort()[0].to(device)
            x = torch.randn(N, self.input_dim, device=device)
            for t in sorted(range(len(sorted_ts)), reverse=True):
                if t == 0:
                    noise = torch.zeros(N, self.input_dim, device=device)
                else:
                    noise = torch.randn(N, self.input_dim, device=device)
                print(
                    t,
                    sorted_ts[t],
                    (beta[t] / torch.sqrt(beta_bar[t])),
                    x,
                    self.forward(x, sorted_ts[t].unsqueeze(0).expand(N)),
                    torch.sqrt(beta[t]) * noise,
                    (1.0 / torch.sqrt(alpha[t])),
                )
                x = (1.0 / torch.sqrt(alpha[t])) * (
                    x
                    - (beta[t] / torch.sqrt(beta_bar[t]))
                    * self.forward(x, sorted_ts[t].unsqueeze(0).expand(N))
                ) + torch.sqrt(beta[t]) * noise
        return x

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class LightningDiffusionModel(pl.LightningModule):
    def __init__(self, input_dim, t_list, hidden_dim):
        super().__init__()
        self.save_hyperparameters()
        self.model = DiffusionModel(input_dim, len(t_list), hidden_dim)
        self.ts = torch.tensor(t_list, dtype=torch.long)

        self._beta = torch.arange(
            0.0, 1.0, 1.0 / (len(t_list) + 1.0), dtype=torch.float32
        )[1:]
        self._alpha = 1.0 - self._beta

        self._alpha_bar = torch.cumprod(self._alpha, dim=0)
        self._beta_bar = 1.0 - self._alpha_bar

    def one_step(self, batch, batch_idx):
        x = batch["x"]
        _ = batch_idx
        _, _, alpha_bar, beta_bar, t = self._get_alpha_beta_t(x)
        noise = torch.randn_like(x)
        x_hat = torch.sqrt(alpha_bar) * x + torch.sqrt(beta_bar) * noise
        score = self.forward(x_hat, t)
        loss_fn = torch.nn.MSELoss(reduction="mean")
        loss = loss_fn(noise, score)
        return loss

    def _get_alpha_beta_t(self, x):
        t_index = torch.randint(len(self.ts), (x.size(0), 1))
        t = self.ts[t_index].to(self.device)
        alpha = self._alpha[t_index].to(self.device)
        beta = self._beta[t_index].to(self.device)
        alpha_bar = self._alpha_bar[t_index].to(self.device)
        beta_bar = self._beta_bar[t_index].to(self.device)
        return alpha, beta, alpha_bar, beta_bar, t.squeeze(1)

    def training_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log(
            "train_loss",
            loss.item(),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, t):
        return self.model(x, t)

    def validation_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss.item(),
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
            loss.item(),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
        )

    def sample(self, N: int):
        return self.model.sample(
            self.ts, self._alpha, self._beta, self._beta_bar, N, device=self.device
        )


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

    model = LightningDiffusionModel(args.input_dim, args.t_list, args.hidden_dim)

    logger = TensorBoardLogger(
        save_dir="data/lightning_logs",
        name="diffusion_model",
    )
    trainer = pl.Trainer(max_epochs=300, logger=logger)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def command_sample(args):
    model = LightningDiffusionModel.load_from_checkpoint(args.ckpt_path)
    model.eval()
    sampled_X = model.sample(N=10000)
    np.savetxt("data/dm_sampled_X.moved.numpy.txt", sampled_X.cpu().detach().numpy())


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("--input_dim", type=int, default=2)
    parser_train.add_argument("--hidden_dim", type=int, default=128)
    parser_train.add_argument(
        "--t_list", nargs="*", type=float, default=[i for i in range(300)]
    )
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
