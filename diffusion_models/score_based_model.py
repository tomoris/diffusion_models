import argparse
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class ScoreBasedModel(torch.nn.Module):
    def __init__(self, input_dim, sigma_num, hidden_dim, sigma_emb_type_linear):
        super().__init__()
        self.input_dim = input_dim
        self.sigma_num = sigma_num
        self.hidden_dim = hidden_dim
        self.sigma_emb_type_linear = sigma_emb_type_linear

        if self.sigma_emb_type_linear:
            self.sigma_embed = torch.nn.Sequential(
                torch.nn.Linear(1, self.hidden_dim),
                torch.nn.Tanh(),
            )
        else:
            self.sigma_embed = torch.nn.Embedding(self.sigma_num, self.hidden_dim)

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

    def forward(self, x, sigma, sigma_index):
        assert sigma.size() == (x.size(0), 1)
        assert sigma_index.size() == (x.size(0),)
        x_embed = self.x_embed(x)

        if not self.sigma_emb_type_linear:
            sigma = sigma.squeeze(1).long()
        sigma_embed = self.sigma_embed(sigma)
        score = self.model(torch.cat([x_embed, sigma_embed], dim=1))
        return score

    def sample(self, sigmas, N, K=5000, alpha=1.0, device="cpu"):
        assert sigmas.dim() == 1
        with torch.no_grad():
            sorted_sigmas = sigmas.sort()[0].to(device)
            x = torch.randn(N, self.input_dim, device=device) * sorted_sigmas[-1]
            for t in sorted(range(len(sorted_sigmas)), reverse=True):
                alpha_t = (
                    alpha
                    * sorted_sigmas[t]
                    * sorted_sigmas[t]
                    / (sorted_sigmas[-1] * sorted_sigmas[-1])
                )
                noise_coeef = torch.sqrt(2 * alpha_t)
                for k in range(K):
                    if t == 0 and k == K - 1:
                        noise = torch.zeros(N, self.input_dim, device=device)
                    else:
                        noise = torch.randn(N, self.input_dim, device=device)
                    tmp_sigma = sorted_sigmas[t].unsqueeze(0).unsqueeze(1).expand(N, 1)
                    tmp_sigma_index = torch.tensor([t for _ in range(N)], device=device)
                    score = self.forward(x, tmp_sigma, tmp_sigma_index)
                    x = x + alpha_t * score + noise_coeef * noise
        return x

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class LightningScoreBasedModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, sigma_list, sigma_emb_type_linear):
        super().__init__()
        self.save_hyperparameters()
        self.model = ScoreBasedModel(
            input_dim, len(sigma_list), hidden_dim, sigma_emb_type_linear
        )
        self.sigmas = torch.tensor(sigma_list, dtype=torch.float32)

    def one_step(self, batch, batch_idx):
        x = batch["x"]
        _ = batch_idx
        sigma_index = torch.randint(len(self.sigmas), (x.size(0), 1))
        sigma = self.sigmas[sigma_index].to(self.device)
        noise = sigma * torch.randn_like(x)
        x_hat = x + noise
        score = self.forward(x_hat, sigma, sigma_index.squeeze(1))
        loss_fn = torch.nn.MSELoss(reduction="none")
        loss = loss_fn(-1.0 * noise, score)
        loss = loss_fn(-1.0 * noise / (sigma * sigma), score)
        loss = sigma * sigma * loss
        loss = loss.sum()
        return loss

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

    def forward(self, x, sigma, sigma_index):
        return self.model(x, sigma, sigma_index)

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
        return self.model.sample(self.sigmas, N, device=self.device)


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

    model = LightningScoreBasedModel(
        args.input_dim, args.hidden_dim, args.sigma_list, args.sigma_emb_type_linear
    )

    logger = TensorBoardLogger(
        save_dir="data/lightning_logs",
        name="score_based_model",
    )
    trainer = pl.Trainer(max_epochs=500, logger=logger)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def command_sample(args):
    model = LightningScoreBasedModel.load_from_checkpoint(args.ckpt_path)
    model.eval()
    sampled_X = model.sample(N=10000)
    np.savetxt("data/sbm_sampled_X.numpy.txt", sampled_X.cpu().detach().numpy())


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train", help="see `train -h`")
    parser_train.add_argument("--input_dim", type=int, default=2)
    parser_train.add_argument("--hidden_dim", type=int, default=128)
    parser_train.add_argument(
        "--sigma_list",
        nargs="*",
        type=float,
        default=[100 * 0.9**i for i in range(100)],
    )
    parser_train.add_argument(
        "--sigma_emb_type_linear", type=int, default=0, help="0: embedding, 1: linear"
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
