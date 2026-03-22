import os
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc
from torch.cuda.amp import autocast
import wandb
import torch.nn as nn
from scheduler import make_scheduler
from utils import CheckpointManager
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

log = logging.getLogger(__name__)

def train_epoch_randcropped(cfg, model, loader, optimizer, transform_train, scheduler, scaler, steps, step_scheduler=False):
    model.train()

    lrs = []
    i = 0
    cumu_loss = 0
    correct, total = (0, 0)
    true_labels = []
    scores = []
    preds = []

    pbar = tqdm(total=steps)

    while True:
        for sample_batched in loader:
            optimizer.zero_grad()

            transform_train(sample_batched)

            # pull the data
            x = sample_batched["signal"]
            age = sample_batched["age"]
            y = sample_batched["class_label"]

            # mixed precision training if needed
            with autocast(enabled=cfg.train.mixed_precision, dtype=torch.float16):
                #forward pass
                output = model(x, age)

                #loss function
                if cfg.train.criterion == "cross-entropy":
                    loss = F.cross_entropy(output, y)

                elif cfg.train.criterion == "multi-bce":
                    y_oh = F.one_hot(y, num_classes=output.size(dim=1))
                    loss = F.binary_cross_entropy_with_logits(output, y_oh.float())
                else:
                    raise ValueError("config['criterion'] must be set to one of ['cross-entropy']")

            # backward and update
            if cfg.train.mixed_precision:
                scaler.scale(loss).backward()
                if cfg.model.grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if step_scheduler:
                    lrs.append(optimizer.param_groups[0]['lr'])
                    scheduler.step()
            else:
                loss.backward()
                if cfg.model.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.grad_clip_norm)
                optimizer.step()
                if step_scheduler:
                    lrs.append(optimizer.param_groups[0]['lr'])
                    scheduler.step()

            # train accuracy
            pred = output.argmax(dim=-1)
            correct += pred.squeeze().eq(y).sum().item()
            total += pred.shape[0]
            cumu_loss += loss.item()

            scores.append(output.detach().cpu())
            preds.append(pred)
            true_labels.append(y)

            i += 1
            pbar.update(1)
            if steps <= i:
                break
        if steps <= i:
            break


    train_acc = 100.0 * correct / total
    avg_loss = cumu_loss / steps 

    return avg_loss, train_acc, torch.cat(true_labels, dim=0), torch.cat(preds, dim=0), torch.cat(scores, dim=0), lrs
    

def test_epoch_windows(cfg, model, test_loader, transform_test): 

    model.eval()

    total_losses = 0.0
    true_labels = []
    scores = []
    preds = []

    with torch.no_grad():

        for batch in test_loader:

            transform_test(batch) 

            x = batch["signal"]
            y = batch["class_label"]
            age = batch["age"]

            output = model(x, age)

            # Loss Calculation
            if cfg.train.criterion == "cross-entropy":
                loss = F.cross_entropy(output, y) 

            elif cfg.train.criterion == "multi-bce": # For the Normal-Dementia-MCI case (3 possible labels)
                y_oh = F.one_hot(y, num_classes=output.size(dim=1))
                loss = F.binary_cross_entropy_with_logits(output, y_oh.float())

            preds.append(output.argmax(dim=1))
            scores.append(output.detach().cpu()) 

            total_losses += loss.item() * x.shape[0]
            true_labels.append(y)
    
    return total_losses, torch.cat(true_labels, dim=0), torch.cat(preds, dim=0), torch.cat(scores, dim=0)


def train_script_window(cfg, model, dataloaders, transform_train, transform_test, do_testing=True):
    
    train_metrics = None
    val_metrics = {
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": []
    }
    test_metrics = {
        "test_loss": [],
        "test_acc": [],
        "test_auc": [],
        "test_f1": []
    }

    lrs = []

    optimizer = optim.AdamW(model.parameters(), lr=cfg.model.base_lr, weight_decay=cfg.model.weight_decay)

    # Get the mixed precision scaler
    amp_scaler = torch.cuda.amp.GradScaler()

    # LR Finder
    if not cfg.train.train_random_crop and cfg.train.find_lr:
        amp_config = {
            'device_type': 'cuda',
            'dtype': torch.float16
        }

        class CustomTrainIter(TrainDataLoaderIter):
            def inputs_labels_from_batch(self, batch_data):
                transform_train(batch_data)
                x = batch_data["signal"]
                y = batch_data["class_label"]
                return x, y
        
        train_iter = CustomTrainIter(dataloaders["train"])

        class CustomValIter(ValDataLoaderIter):
            def inputs_labels_from_batch(self, batch_data):
                transform_test(batch_data)
                x = batch_data["signal"]
                y = batch_data["class_label"]
                return x, y

        val_iter = CustomValIter(dataloaders["validation"])

        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cuda",
            amp_backend='torch',
            amp_config=amp_config,
            grad_scaler=amp_scaler
        )

        lr_finder.range_test(train_iter, val_loader=val_iter, end_lr=0.1, num_iter=100, step_mode="exp")
        lr_finder.plot()

    # Initialize the scheduler
    scheduler = make_scheduler(cfg, optimizer)

    # Init the checkpointer
    checkpoint = CheckpointManager(
        base_epochs=0, # Number of epochs that need to elapse before the checkpointer starts the patience countdown
        patience=cfg.model.patience,
        model_name=cfg.model.model_name,
        save_path=cfg.output_dir
    )

    elapsed_epochs = 0
    num_training_steps = -(int(cfg.train.samples_per_epoch) // -int(cfg.model.minibatch)) # neg. to convert to ceiling division to round up remainder
    for epoch in range(cfg.model.epochs):
        
        log.info(f"Starting Epoch {epoch}.")
        print(f"Learning Rate for epoch {epoch}: {optimizer.param_groups[0]['lr']}")

        # Train ##########################################
        if not cfg.train.train_random_crop:
            raise ValueError("Fixed Window training is not supported in this version of the codebase.")
        else:
            total_train_loss, train_acc, train_labels, train_preds, train_scores, epoch_lrs = train_epoch_randcropped(
                cfg, 
                model,
                dataloaders["train"],
                optimizer,
                transform_train=transform_train,
                scheduler=scheduler, 
                scaler=amp_scaler,
                steps=num_training_steps,
                step_scheduler=cfg.model.step_scheduler_batch
            )

            if scheduler is not None and not cfg.model.step_scheduler_batch: 
                lrs.append(optimizer.param_groups[0]['lr'])
                scheduler.step(epoch)
            else:
                lrs.extend(epoch_lrs)

        train_correct = train_preds.squeeze().eq(train_labels).sum().item()
        train_acc = 100 * train_correct/len(train_labels)

        if cfg.model.out_dims == 2:
            train_auc = binary_auroc(preds=train_scores.detach().cpu(), target=train_labels.detach().cpu())
            log.info(f"Train loss: {total_train_loss} - Train acc {train_acc} - Train AUROC {train_auc}")

        elif cfg.model.out_dims == 3:
            train_auc = multiclass_auroc(preds=train_scores.detach().cpu().float(), target=train_labels.detach().cpu(), num_classes=3, average=None)
            avg_train_auc = torch.sum(train_auc) / 3
            train_f1 = f1_score(train_labels.detach().cpu(), train_preds.detach().cpu(), average=None)
            log.info(f"Train loss: {total_train_loss} - Train acc {train_acc} - Train AUROC Class: {train_auc} - Train AUROC {avg_train_auc} - Train F1: {train_f1}")
        else:
            raise ValueError(f"Out dims is a corrupted number: {cfg.model.out_dims}")        

        #############################################################

        # Validation ################################################
        val_loss, val_labels, val_preds, val_scores = test_epoch_windows(
            cfg,
            model,
            dataloaders["validation"],
            transform_test
        )
    
        val_loss = val_loss / len(val_labels)
        val_correct = val_preds.squeeze().eq(val_labels).sum().item()
        val_acc = 100 * val_correct/len(val_labels)
        
        if cfg.model.out_dims == 2:
            val_auc = binary_auroc(preds=val_scores.detach().cpu(), target=val_labels.detach().cpu())
            log.info(f"Val loss: {val_loss} - Val acc: {val_acc} - Val AUROC:{val_auc}")

        elif cfg.model.out_dims == 3:
            val_auc = multiclass_auroc(preds=val_scores.detach().cpu(), target=val_labels.detach().cpu(), num_classes=3, average=None)
            avg_val_auc = torch.sum(val_auc) / 3
            val_f1 = f1_score(val_labels.detach().cpu(), val_preds.detach().cpu(), average=None)
            log.info(f"Val loss: {val_loss} - Val acc: {val_acc} - Val AUROC Class: {val_auc} - Val AUROC Avg.:{avg_val_auc} - Val F1: {val_f1}")     

        val_metrics["val_loss"].append(val_loss)
        val_metrics["val_acc"].append(val_acc)
        val_metrics["val_auc"].append(avg_val_auc.item())
        
        # ##############################################################

        if cfg.train.use_wandb:
            wandb.log(
                {"train/epoch_loss": total_train_loss,
                "train/epoch_acc": train_acc,
                "val/epoch_loss": val_loss,
                "val/epoch_acc": val_acc,
                "LR": lrs[-1],
                "Elapsed Epochs": epoch
            })


    if cfg.train.plot_lr:
        plt.plot([i for i in range(len(lrs))], lrs)
        savepath = os.path.join(str(cfg.output_dir), 'lrs.png') 
        plt.savefig(savepath)

    if do_testing:

        test_loss, test_labels, test_preds, test_scores = test_epoch_windows(
            cfg,
            model,
            dataloaders["test"],
            transform_test
        )
    
        test_loss = test_loss / len(test_labels)
        test_correct = test_preds.squeeze().eq(test_labels).sum().item()
        test_acc = 100 * test_correct/len(test_labels)
        
        if cfg.model.out_dims == 2:
            test_auc = binary_auroc(preds=test_scores.detach().cpu(), target=test_labels.detach().cpu())
            log.info(f"test loss: {test_loss} - test acc: {test_acc} - test AUROC:{test_auc}")

        elif cfg.model.out_dims == 3:
            test_auc = multiclass_auroc(preds=test_scores.detach().cpu(), target=test_labels.detach().cpu(), num_classes=3, average=None)
            avg_test_auc = torch.sum(test_auc) / 3
            test_f1 = f1_score(test_labels.detach().cpu(), test_preds.detach().cpu(), average=None)
            log.info(f"test loss: {test_loss} - test acc: {test_acc} - test AUROC Class: {test_auc} - test AUROC Avg.:{avg_test_auc} - test F1: {test_f1}")     
        
        # Save the model
        checkpoint.save_overtrained_model(model, optimizer=optimizer)

        test_metrics["test_loss"].append(test_loss)
        test_metrics["test_acc"].append(test_acc)
        test_metrics["test_auc"].append(avg_test_auc.item())
        ############################################################

    # Delete old model versions to save space
    checkpoint.delete_old_models()

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "lrs": lrs
    }

    return metrics
