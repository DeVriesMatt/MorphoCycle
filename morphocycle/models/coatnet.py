import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall, MetricCollection
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import wandb



def create_model(model_name="IMAGENET1K_V2", pretrained=True, num_classes=8, **kargs):
    # model = timm.create_model(model_name, pretrained=pretrained)

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # transforms = timm.data.create_transform(**data_config, is_training=False)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
    for param in model.parameters():
        param.requires_grad = True

    # num_features = model.classifier.fc.in_features
    # model.classifier.fc = nn.Linear(num_features, num_classes)
    # model.classifier.fc.requires_grad = True
    #
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.fc.requires_grad = True

    return model


class Classifier(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        prob_transform=0.5,
        max_epochs=250,
        log_dir="./logs",
        model_name='coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k',
        pretrained=True,
        learning_rate=0.00001,


        **kwargs,
    ):
        super(Classifier, self).__init__()

        self.save_hyperparameters(ignore=["criterion"])
        self.lr = learning_rate
        self.criterion = criterion

        self.model = create_model(model_name=model_name,
                                                   pretrained=pretrained,
                                                   num_classes=num_classes,
                                                   **kwargs)

        self.num_classes = num_classes
        self.prob_transform = prob_transform
        self.max_epochs = max_epochs
        if num_classes > 2:
            self.acc = Accuracy(task="multiclass", average="macro", num_classes=num_classes)
            self.auc = AUROC(task="multiclass", num_classes=num_classes, average="macro")
            self.F1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.precision_metric = Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        else:
            self.acc = Accuracy(task="binary", average="macro")
            self.auc = AUROC(task="binary", num_classes=num_classes, average="macro")
            self.F1 = F1Score(task="binary", num_classes=num_classes, average="macro")
            self.precision_metric = Precision(
                task="binary", num_classes=num_classes, average="macro"
            )
            self.recall = Recall(task="binary", num_classes=num_classes, average="macro")


        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        self.log_path = log_dir
        metrics = MetricCollection(
            [
                self.acc,
                self.F1,
                self.precision_metric,
                self.recall,
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        # )
        return [optimizer]  # , [lr_scheduler]

    def calculate_loss(self, inputs, labels):
        logits = self.forward(inputs)
        loss = self.criterion(logits, labels.long())
        y_prob = torch.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)
        return loss, y_prob, y_hat, logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1].double()
        loss, y_prob, y_hat, logits = self.calculate_loss(inputs, labels)
        acc = self.acc(y_hat, labels)
        #
        # sample_images = inputs[:6]
        # grid = torchvision.utils.make_grid(sample_images)
        # captions = [f'Truth: {y_i} - Prediction: {y_pred}'
        #             for y_i, y_pred in zip(labels[:6], y_hat[:6])]
        # self.logger.experiment.log({"images": wandb.Image(grid.cpu(), caption=captions)})
        #
        # class_names = ['G1', 'S', 'G2', 'MorG1']
        # self.logger.experiment.log({"conf_mat" : wandb.plot.confusion_matrix(
        #     probs=None,
        #     y_true=torch.squeeze(labels).detach().cpu().numpy(),
        #     preds=torch.squeeze(y_hat).detach().cpu().numpy(),
        #     class_names=class_names)}
        # )

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )


        # self.data[int(labels)]["count"] += 1
        # self.data[int(labels)]["correct"] += y_hat == labels

        dic = {
            "loss": loss,
            "acc": acc,
        }
        return dic

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1].double()
        loss, y_prob, y_hat, logits = self.calculate_loss(inputs, labels)
        acc = self.acc(y_hat, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        # sample_images = inputs[:6]
        # grid = torchvision.utils.make_grid(sample_images)
        # captions = [f'Truth: {y_i} - Prediction: {y_pred}'
        #             for y_i, y_pred in zip(labels[:6], y_hat[:6])]
        # self.logger.experiment.log({"images_val": wandb.Image(grid.cpu(), caption=captions)})


        # ---->acc log
        # self.data[int(labels)]["count"] += 1
        # self.data[int(labels)]["correct"] += y_hat == labels


        results = {
            "logits": logits,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": labels,
        }
        # print(y_hat == labels)
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs], dim=0)
        probs = torch.cat([x["Y_prob"] for x in self.validation_step_outputs], dim=0)
        max_probs = torch.cat([x["Y_hat"] for x in self.validation_step_outputs], dim=0)
        target = torch.cat([x["label"] for x in self.validation_step_outputs], dim=0)

        # ---->
        self.log(
            "val_loss",
            self.criterion(logits, target.squeeze().long()),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "auc",
            self.auc(probs, target.squeeze().long()),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log_dict(
            self.valid_metrics(max_probs.squeeze(), target.squeeze()),
            on_epoch=True,
            logger=True,
        )
        # ---->acc log
        # for c in range(self.num_classes):
        #     count = self.data[c]["count"]
        #     correct = self.data[c]["correct"]
        #     if count == 0:
        #         acc = None
        #     else:
        #         acc = float(correct) / count
        #     print("class {}: acc {}, correct {}/{}".format(c, acc, correct.item(), count))
        # self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1].double()
        loss, y_prob, y_hat, logits = self.calculate_loss(inputs, labels)
        acc = self.acc(y_hat, labels)

        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        # self.data[int(labels)]["count"] += 1

        # self.data[int(labels)]["correct"] += y_hat == labels

        results = {
            "logits": logits,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": labels,
        }
        self.test_step_outputs.append(results)

        return results

    def on_test_end(self):
        probs = torch.cat([x["Y_prob"] for x in self.test_step_outputs], dim=0)
        max_probs = torch.cat([x["Y_hat"] for x in self.test_step_outputs], dim=0)
        target = torch.cat([x["label"] for x in self.test_step_outputs], dim=0)

        # ---->
        auc = self.auc(probs, target.squeeze().long())
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics["auc"] = auc
        for keys, values in metrics.items():
            print(f"{keys} = {values}")
            metrics[keys] = values.cpu().numpy()
        print()

        class_names = ['G1', 'S', 'G2', 'M']
        self.logger.experiment.log({"conf_mat_val": wandb.plot.confusion_matrix(
            probs=None,
            y_true=torch.squeeze(target).detach().cpu().numpy(),
            preds=torch.squeeze(max_probs).detach().cpu().numpy(),
            class_names=class_names)}
        )
        # ---->acc log
        # for c in range(self.num_classes):
        #     count = self.data[c]["count"]
        #     correct = self.data[c]["correct"]
        #     print("val count", count)
        #     if count == 0:
        #         acc = None
        #     else:
        #         acc = float(correct) / count
        #     print("class {}: acc {}, correct {}/{}".format(c, acc, correct.item(), count))
        # self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        # ---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path + "/result.csv")

        self.test_step_outputs.clear()