import lightning as pl
from models.unet import UNet
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch import nn
from torchvision.utils import make_grid
import wandb
import timm


class Pix2Pix(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.generator = UNet(num_classes=1)
        self.discriminator = timm.create_model('resnet18', in_chans=2, pretrained=True)

        for param in self.discriminator.parameters():
            param.requires_grad = True


        # # num_features = model.classifier.fc.in_features
        # # model.classifier.fc = nn.Linear(num_features, num_classes)
        # # model.classifier.fc.requires_grad = True
        # #
        num_features = self.discriminator.fc.in_features
        self.discriminator.fc = nn.Linear(num_features, 1)
        self.discriminator.fc.requires_grad = True

        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.automatic_optimization = False

        # Optional: define other loss functions like perceptual loss

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999

        # Optimizers for generator and discriminator
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr/10, betas=(b1, b2))
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )

        return [opt_gen, opt_disc], []

    def forward(self, x):
        # Forward pass through the generator (U-Net)
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        input_images, target_images = batch["input"], batch["target"]

        optimizer_g, optimizer_d = self.optimizers()

        # Generator forward pass
        self.toggle_optimizer(optimizer_g)
        generated_imgs = self(input_images)

        content_loss = self.l1_loss(generated_imgs, target_images)
        self.log("content_loss", content_loss, prog_bar=True)

        # Concatenate input with real and fake images for the discriminator
        real_pairs = torch.cat((input_images, target_images), dim=1)
        fake_pairs = torch.cat((input_images, generated_imgs), dim=1)

        # Add noise to real pairs for discriminator training
        noise_std = 0.05
        noise = torch.randn_like(real_pairs) * noise_std
        noisy_real_pairs = real_pairs + noise

        # Generate smooth labels for real and fake
        smooth_real_labels = torch.FloatTensor(target_images.size(0), 1).uniform_(0.8, 0.9).type_as(target_images)
        smooth_fake_labels = torch.FloatTensor(target_images.size(0), 1).uniform_(0.0, 0.1).type_as(target_images)

        # Generator's adversarial loss
        g_loss = self.adversarial_loss(self.discriminator(fake_pairs), smooth_real_labels) + content_loss
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        self.toggle_optimizer(optimizer_d)

        # Real loss
        real_loss = self.adversarial_loss(self.discriminator(noisy_real_pairs), smooth_real_labels)

        # Fake loss
        fake_loss = self.adversarial_loss(self.discriminator(fake_pairs.detach()), smooth_fake_labels)

        # Discriminator loss is the average of real and fake losses
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        input_images, target_images = batch["input"], batch["target"]

        # Forward pass
        generated_images = self(input_images)

        # Calculate loss (you can use the same losses as in training or different ones)
        val_loss = self.l1_loss(generated_images, target_images)  # Example: L1 loss

        # Log the validation loss
        self.log("val_loss", val_loss)

        log_images = batch_idx == 0  # Example: log only the first batch of each epoch

        grid_input = make_grid(input_images, normalize=True)
        grid_target = make_grid(target_images, normalize=True)
        grid_gen = make_grid(generated_images, normalize=True)

        self.logger.experiment.log(
            {
                "phase_images": wandb.Image(grid_input.cpu()),
                "fluorescent_images": wandb.Image(grid_target.cpu()),
                "generated_fluorescent_images": wandb.Image(grid_gen.cpu()),
            }
        )

        return {
            "phase_images": input_images,
            "fluorescent_images": target_images,
            "generated_images": generated_images,
            "log_images": log_images,
        }