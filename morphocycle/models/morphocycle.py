import lightning as pl
from models.unet import UNet
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
from torchvision.utils import make_grid
import wandb


class MorphoCycle(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.generator = UNet(num_classes=3)
        self.discriminator = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.discriminator.parameters():
            param.requires_grad = True

        # num_features = model.classifier.fc.in_features
        # model.classifier.fc = nn.Linear(num_features, num_classes)
        # model.classifier.fc.requires_grad = True
        #
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
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr/10, betas=(b1, b2)
        )

        return [opt_gen, opt_disc], []

    def forward(self, x):
        # Forward pass through the generator (U-Net)
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        input_images, target_images = batch["input"], batch["target"]

        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(input_images)

        content_loss = self.l1_loss(self.generated_imgs, target_images)
        self.log("content_loss", content_loss, prog_bar=True)

        # log sampled images
        sample_gen_imgs = self.generated_imgs[:1]
        grid_gen = make_grid(sample_gen_imgs)


        sample_input_imgs = input_images[:1]
        grid_input = make_grid(sample_input_imgs)


        sample_target_imgs = target_images[:1]
        grid_target = make_grid(sample_target_imgs)


        self.logger.experiment.log(
            {
                "phase_images": wandb.Image(grid_input.cpu()),
                "fluorescent_images": wandb.Image(grid_target.cpu()),
                "generated_fluorescent_images": wandb.Image(grid_gen.cpu()),
            }
        )

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(self.generated_imgs.size(0), 1)
        valid = valid.type_as(self.generated_imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(input_images)), valid) + content_loss
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(target_images.size(0), 1)
        valid = valid.type_as(target_images)

        real_loss = self.adversarial_loss(self.discriminator(target_images), valid)

        # how well can it label as fake?
        fake = torch.zeros(target_images.size(0), 1)
        fake = fake.type_as(target_images)

        fake_loss = self.adversarial_loss(self.discriminator(self(input_images).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)


        # # Train generator
        # if optimizer_idx == 0:
        #     # Generate images
        #     generated_images = self(real_images)
        #
        #     # Calculate adversarial loss (generator wants to fool discriminator)
        #     gen_discriminated = self.discriminator(generated_images)
        #     valid = torch.ones_like(gen_discriminated)
        #     gen_adv_loss = self.adversarial_loss(gen_discriminated, valid)
        #
        #     # Calculate content loss (e.g., L1 loss)
        #     content_loss = self.l1_loss(generated_images, target_images)
        #
        #     # Optional: calculate other losses (perceptual, SSIM, etc.)
        #
        #     # Combine losses for generator
        #     g_loss = gen_adv_loss + content_loss  # Add other losses as needed
        #     self.log("generator_loss", g_loss, on_step=True, on_epoch=True, logger=True)
        #     return g_loss
        #
        # # Train discriminator
        # if optimizer_idx == 1:
        #     # Real loss
        #     real_discriminated = self.discriminator(target_images)
        #     valid = torch.ones_like(real_discriminated)
        #     real_loss = self.adversarial_loss(real_discriminated, valid)
        #
        #     # Fake loss
        #     generated_images = self(real_images).detach()
        #     fake_discriminated = self.discriminator(generated_images)
        #     fake = torch.zeros_like(fake_discriminated)
        #     fake_loss = self.adversarial_loss(fake_discriminated, fake)
        #
        #     # Combine losses for discriminator
        #     d_loss = (real_loss + fake_loss) / 2
        #     self.log(
        #         "discriminator_loss", d_loss, on_step=True, on_epoch=True, logger=True
        #     )
        #     return d_loss

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
