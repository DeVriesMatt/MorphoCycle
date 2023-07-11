import timm
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall, MetricCollection
import pandas as pd


def create_model(model_name='coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k', pretrained=True, num_classes=8, **kargs):
    model = timm.create_model(model_name, pretrained=pretrained)

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.head.fc.in_features
    model.head.fc = nn.Linear(num_features, num_classes)
    model.head.fc.requires_grad = True

    return model, transforms


class COATNet(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        prob_transform=0.5,
        max_epochs=250,
        log_dir="./logs",
        model_name='coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k',
        pretrained=True,


        **kwargs,
    ):
        super(COATNet, self).__init__()

        self.save_hyperparameters(ignore=["criterion"])
        self.lr = 0.00001
        self.criterion = criterion

        self.model, self.transforms = create_model(model_name=model_name,
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def calculate_loss(self, inputs, labels):
        logits = self.forward(inputs)
        loss = self.criterion(logits, labels.long())
        y_prob = torch.softmax(logits, dim=1)[:, 1]
        y_hat = torch.argmax(y_prob, dim=1)
        return loss, y_prob, y_hat, logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        loss, y_prob, y_hat = self.calculate_loss(inputs, labels)
        acc = self.acc(y_prob, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        Y_hat_c = int(y_prob > 0.5)
        Y_c = int(labels.unsqueeze(1))
        self.data[Y_c]["count"] += 1
        self.data[Y_c]["correct"] += Y_hat_c == Y_c

        dic = {
            "loss": loss,
            "acc": acc,
        }
        return dic

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        loss, y_prob, y_hat, logits = self.calculate_loss(inputs, labels)
        acc = self.acc(y_prob, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        y = int(labels)
        self.data[y]["count"] += 1
        self.data[y]["correct"] += y_hat == y

        results = {
            "logits": logits,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": labels,
        }
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs], dim=0)
        probs = torch.cat([x["y_prob"] for x in self.validation_step_outputs], dim=0)
        max_probs = torch.stack([x["y_hat"] for x in self.validation_step_outputs])
        target = torch.stack([x["label"] for x in self.validation_step_outputs], dim=0)

        # ---->
        self.log(
            "val_loss",
            self.criterion(logits, target),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "auc",
            self.auc(probs, target.squeeze()),
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
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print("class {}: acc {}, correct {}/{}".format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        loss, y_prob, y_hat, logits = self.calculate_loss(inputs, labels)
        acc = self.acc(y_prob, labels.unsqueeze(1))

        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        y = int(labels)
        self.data[y]["count"] += 1
        self.data[y]["correct"] += y_hat == y

        results = {
            "logits": logits,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": labels,
        }
        self.test_step_outputs.append(results)

        return results

    def on_test_end(self):
        probs = torch.cat([x["y_prob"] for x in self.test_step_outputs], dim=0)
        max_probs = torch.stack([x["y_hat"] for x in self.test_step_outputs])
        target = torch.stack([x["label"] for x in self.test_step_outputs], dim=0)

        # ---->
        auc = self.auc(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics["auc"] = auc
        for keys, values in metrics.items():
            print(f"{keys} = {values}")
            metrics[keys] = values.cpu().numpy()
        print()
        # ---->acc log
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            print("val count", count)
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print("class {}: acc {}, correct {}/{}".format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        # ---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path + "/result.csv")

        self.test_step_outputs.clear()