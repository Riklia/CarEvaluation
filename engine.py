import os
import torch
from tqdm.auto import tqdm
import numpy as np
import logging
import utils
from early_stopping import EarlyStopping

# early_stopping = EarlyStopping(patience=25, mode="max", delta=0.001)
early_stopping = None


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model using batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_bloking=True), labels_batch.cuda(non_bloking=True)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            parameters = []
            for parameter in model.parameters():
                parameters.append(parameter.view(-1))
            l1 = params.l1_weight * model.compute_l1_loss(torch.cat(parameters))
            # l2 = params.l2_weight * model.compute_l2_loss(torch.cat(parameters))
            loss += l1
            # loss += l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % params.save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy().argmax(
                    axis=1)
                labels_batch = labels_batch.data.cpu().numpy().argmax(
                    axis=1)
                summary_batch = {
                    metric: metrics[metric](output_batch, labels_batch)
                    for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)

        return metrics_mean


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model using batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    model.eval()
    summ = []
    loss_avg = utils.RunningAverage()
    for data_batch, labels_batch in dataloader:
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))
        l1 = params.l1_weight * model.compute_l1_loss(torch.cat(parameters))
        # l2 = params.l2_weight * model.compute_l2_loss(torch.cat(parameters))
        loss += l1
        # loss += l2
        loss_avg.update(loss.item())
        output_batch = output_batch.data.cpu().numpy().argmax(axis=1)
        labels_batch = labels_batch.data.cpu().numpy().argmax(axis=1)
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in
                    summ[0]}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer,
                       loss_fn,
                       metrics, params, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (generator) a generator that generates batches of data and labels
        test_dataloader: (generator) a generator that generates batches of data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        restore_file: (string) optional - name of file to restore from (
        without its extension .pth.tar)
    """

    if restore_file is not None:
        restore_path = os.path.join(
            params.model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    train_stats = []
    val_stats = []

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        train_metrics = train(model, optimizer, loss_fn, train_dataloader,
                              metrics, params)
        val_metrics = evaluate(
            model, loss_fn, test_dataloader, metrics, params)

        val_f1_macro = val_metrics['f1_score_macro']
        is_best = val_f1_macro >= best_val_acc
        utils.save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=params.model_dir)

        if early_stopping:
            early_stopping(val_f1_macro, model, os.path.join(
                params.model_dir, "metrics_val_last_weights.json"))

            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_f1_macro
            best_json_path = os.path.join(
                params.model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(
            params.model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        train_stats.append(train_metrics)
        val_stats.append(val_metrics)

    return train_stats, val_stats
