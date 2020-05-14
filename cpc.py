import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from argparse import ArgumentParser

import math


class CPCV2(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # encoder network (Z vectors)
        dummy_batch = torch.zeros((2, 3, hparams.patch_size, hparams.patch_size))
        self.encoder = CPCResNet101(dummy_batch)

        # context network (C vectors)
        c, h = self.__compute_final_nb_c(hparams.patch_size)
        self.context_network = PixelCNN(c)

        self.target_dim = 64
        self.target_cnn = torch.nn.Conv2d(c, self.target_dim, kernel_size=1)
        self.info_nce_pred_cnn = torch.nn.Conv2d(c, self.target_dim, kernel_size=1)

        self.tng_split = None
        self.val_split = None

    def __compute_final_nb_c(self, patch_size):
        dummy_batch = torch.zeros((2*49, 3, patch_size, patch_size))
        dummy_batch = self.encoder(dummy_batch)
        dummy_batch = self.__recover_z_shape(dummy_batch, 2)
        b, c, h, w = dummy_batch.size()
        return c, h

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_feats = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_feats, nb_feats)

        return Z

    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, p, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        img_1, _ = batch

        # Latent features
        Z = self.forward(img_1)

        # infoNCE loss
        loss = self.info_nce_loss(Z)

        result = {
            'loss': loss
        }

        return result

    def info_nce_loss(self, Z, target_dim=64, emb_scale=0.1, steps_to_ignore=2, steps_to_predict= 3):
        loss = 0.0

        # generate the context vars
        C = self.context_network(Z)

        # generate targets
        targets = self.target_cnn(C)
        batch_dim, _, col_dim, row_dim = targets.shape
        targets = targets.reshape(-1, target_dim)

        for i in range(steps_to_ignore, steps_to_predict):
            col_dim_i = col_dim - i - 1
            total_elements = batch_dim * col_dim_i * row_dim

            preds_i = self.info_nce_pred_cnn(C)
            preds_i = preds_i[:, :, :-(i + 1), :] * emb_scale
            preds_i = preds_i.reshape(-1, target_dim)

            logits = torch.mm(preds_i, targets.transpose(1, 0))
            b = torch.range(0, total_elements-1) / (col_dim_i * row_dim)
            col = torch.range(0, total_elements-1) % (col_dim_i * row_dim)
            labels = b * col_dim * row_dim + (i + 1) * row_dim + col

            logits = F.log_softmax(logits, dim=1)
            nll = F.nll_loss(logits, labels) # < ----------------- FAIL HERE labels has a range [0, 134]
            loss += nll.mean()

        return loss


    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-7
        )

        lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)
        return [opt], [lr_scheduler]

    def train_dataloader(self):
        train_transform = cpc_transforms.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
        dataset = UnlabeledImagenet(self.hparams.data_dir,
                                    nb_classes=self.hparams.nb_classes,
                                    split='train',
                                    transform=train_transform)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=16,
        )

        return loader


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = CPCV2(args)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
