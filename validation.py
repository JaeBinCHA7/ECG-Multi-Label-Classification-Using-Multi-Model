import argparse
import torch
import options
import utils
import random
import numpy as np
from sklearn.metrics import accuracy_score
import sklearn

######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
# Parse command-line arguments and configurations for the experiment
opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification')).parse_args()
print(opt)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# Set the device (either CPU or GPU) and initialize the random seeds for reproducibility
DEVICE = torch.device(opt.device)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

model, dataloader, pretrained_model, class_num = utils.get_valid_mode(opt)

total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# Load a pre-trained model
print('Load the pretrained model...')
chkpt1 = torch.load(pretrained_model)
model.load_state_dict(chkpt1['model'])
model = model.to(DEVICE)
valid_loader = dataloader(opt)

######################################################################################################################
#                                             Main program - test                                                    #
######################################################################################################################
print('Validation start...')

t_all = []
b_all = []
p_all = []
o_all = []
dx_all = []
dx_mismatch_t1_b0 = []  # Case where t_all is 1 and b_all is 0
dx_mismatch_t0_b1 = []  # Case where t_all is 0 and b_all is 1
threshold = 0.5
# test
model.eval()
with torch.no_grad():
    for inputs, targets in utils.Bar(valid_loader):
        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        # model
        logit = model(inputs)
        p_out = torch.sigmoid(logit)
        b_out = p_out > torch.tensor(threshold, device=DEVICE)

        b_all.append(b_out.data.cpu().numpy())
        p_all.append(p_out.data.cpu().numpy())
        t_all.append(targets.data.cpu().numpy())

    ######################################################################################################################
    #                                                   Get scores                                                       #
    ######################################################################################################################
    t_all = np.concatenate(t_all, axis=0)
    b_all = np.concatenate(b_all, axis=0)
    p_all = np.concatenate(p_all, axis=0)

    # Compute evaluation metrics (AUROC, AUPRC, F-Measure, and Kappa) for the predictions
    s_all = np.zeros((len(b_all), class_num), dtype=np.float64)
    for i in range(len(b_all)):
        s_all[i] = [True if v == 1 else False for v in b_all[i]]

    auroc, auprc, auroc_classes, auprc_classes = utils.compute_auc(t_all, p_all)
    f_measure, f1score_classes, precision, _, recall, _, specificity, _ = utils.compute_f_measure_and_precision_recall_specificity(
        t_all, b_all)
    kappa = utils.compute_kappa(t_all, b_all, num_classes=class_num)
    hamming_loss = sklearn.metrics.hamming_loss(t_all, b_all)

print("####################################################################")
print(pretrained_model)
print("Kappa : {:.4}".format(kappa))
print("AUROC : {:.4}".format(auroc))
print("precision : {:.4}".format(precision))
print("recall : {:.4}".format(recall))
print("specificity : {:.4}".format(specificity))
print("AUPRC : {:.4}".format(auprc))
print("F-Measure : {:.4}".format(f_measure))
print("Hamming loss : {:.4}".format(hamming_loss))
print("####################################################################")

print('System has been finished.')
