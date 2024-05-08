import torch
from .data_utils import Bar, np
from sklearn.metrics import average_precision_score, accuracy_score
from utils import evaluate, visualize_pair_results
import utils


######################################################################################################################
#    Multi-label classification
######################################################################################################################
# Define the training function for the model.
def ms_train(model, train_loader, loss_calculator, optimizer, writer,
             EPOCH, DEVICE):
    # Initialization of variables to track training loss and number of batches.
    train_loss = 0
    batch_num = 0

    # Train
    model.train()

    # Iterate through batches of data from the train_loader.
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        outputs = np.squeeze(outputs)

        loss = loss_calculator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Update model parameters based on gradients.

        train_loss += loss.item()
    train_loss /= batch_num

    # Log the average training loss to tensorboard.
    writer.log_train_loss('total', train_loss, EPOCH)

    return train_loss


def ms_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE):
    # Initialization of variables to track validation loss and number of batches.
    valid_loss = 0
    batch_num = 0

    t_all = []
    p_all = []

    # Validation
    model.eval()

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)
            outputs = np.squeeze(outputs)

            p = torch.sigmoid(outputs)
            loss = loss_calculator(outputs, targets)

            valid_loss += loss

            t_all.append(targets.data.cpu().numpy())
            p_all.append(p.data.cpu().numpy())
        valid_loss /= batch_num

    t_all = np.concatenate(t_all, axis=0)
    p_all = np.concatenate(p_all, axis=0)
    b_all = p_all > 0.5
    # Compute the Area Under the Precision-Recall Curve (AUPRC).
    valid_auprc = average_precision_score(y_true=t_all, y_score=p_all)
    valid_f1, _, _, _, _, _, _, _ = utils.compute_f_measure_and_precision_recall_specificity(
        t_all, b_all)

    # Log the average validation loss to tensorboard.
    writer.log_valid_loss('total', valid_loss, EPOCH)
    writer.log_score('AUPRC', valid_auprc, EPOCH)
    writer.log_score('F1-score', valid_f1, EPOCH)

    return valid_loss, valid_auprc, valid_f1
