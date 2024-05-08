import os
import argparse
import torch
import options
import utils
import datetime
import random
import numpy as np
from dataloader import dataloader_inference
from sklearn.metrics import accuracy_score
from models import ResU_Dense
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

# Initialize the model and print its size
model = utils.get_inferece_mode(opt)
model_rhythm = model[0]
model_duration = model[1]
model_amplitude = model[2]
model_morphology = model[3]

total_params = utils.cal_total_params(model[0]) * 4
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# Load a pre-trained model
print('Load the pretrained model...')
chkpt1 = torch.load(opt.model_rhythm)
model_rhythm.load_state_dict(chkpt1['model'])
model_rhythm = model_rhythm.to(DEVICE)

chkpt2 = torch.load(opt.pretrained_model_duration)
model_duration.load_state_dict(chkpt2['model'])
model_duration = model_duration.to(DEVICE)

chkpt3 = torch.load(opt.pretrained_model_amplitude)
model_amplitude.load_state_dict(chkpt3['model'])
model_amplitude = model_amplitude.to(DEVICE)

chkpt4 = torch.load(opt.pretrained_model_morphology)
model_morphology.load_state_dict(chkpt4['model'])
model_morphology = model_morphology.to(DEVICE)
######################################################################################################################
#                                             Main program - test                                                    #
######################################################################################################################
# Start testing
test_loader = dataloader_inference(opt)

print('Test start...')

t_all = []
b_all = []
p_all = []
o_all = []

target_labels = np.load(opt.target_all, allow_pickle=True)
labels_hr = np.load(opt.target_rhythm, allow_pickle=True)
labels_iv = np.load(opt.target_duration, allow_pickle=True)
labels_ap = np.load(opt.target_amplitude, allow_pickle=True)
labels_mp = np.load(opt.target_morphology, allow_pickle=True)

threshold = 0.5

# test
model_rhythm.eval()
model_duration.eval()
model_amplitude.eval()
model_morphology.eval()
with torch.no_grad():
    for inputs, targets in utils.Bar(test_loader):
        b_out = np.zeros(len(target_labels), dtype=int)
        p_out = np.zeros(len(target_labels))

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        # Regularity
        out_r = model_rhythm(inputs)
        p_r = torch.sigmoid(out_r)
        b_r = p_r > torch.tensor(threshold, device=DEVICE)

        for i, label in enumerate(labels_hr):
            index_in_target = np.where(target_labels == label)[0][0]  # Find the matching index in target_labels
            b_out[index_in_target] = b_r[:, i]
            p_out[index_in_target] = p_r[:, i]

        # Irregularity
        out_i = model_duration(inputs)
        p_i = torch.sigmoid(out_i)
        b_i = p_i > torch.tensor(threshold, device=DEVICE)

        for i, label in enumerate(labels_iv):
            index_in_target = np.where(target_labels == label)[0][0]  # Find the matching index in target_labels
            b_out[index_in_target] = b_i[:, i]
            p_out[index_in_target] = p_i[:, i]

        # Morphology
        out_c = model_amplitude(inputs)
        p_c = torch.sigmoid(out_c)
        b_c = p_c > torch.tensor(threshold, device=DEVICE)

        for i, label in enumerate(labels_ap):
            index_in_target = np.where(target_labels == label)[0][0]  # Find the matching index in target_labels
            b_out[index_in_target] = b_c[:, i]
            p_out[index_in_target] = p_c[:, i]

        # Scope
        out_h = model_morphology(inputs)
        p_h = torch.sigmoid(out_h)
        b_h = p_h > torch.tensor(threshold, device=DEVICE)

        for i, label in enumerate(labels_mp):
            index_in_target = np.where(target_labels == label)[0][0]  # Find the matching index in target_labels
            b_out[index_in_target] = b_h[:, i]
            p_out[index_in_target] = p_h[:, i]

        b_all.append(b_out)
        p_all.append(p_out)
        t_all.append(targets.data.cpu().numpy())

    ######################################################################################################################
    #                                                   Get scores                                                       #
    ######################################################################################################################
    t_all = np.concatenate(t_all, axis=0)
    b_all = np.array(b_all)
    p_all = np.array(p_all)

    # Compute evaluation metrics (AUROC, AUPRC, F-Measure, and Kappa) for the predictions
    num_classes = opt.classes

    s_all = np.zeros((len(b_all), num_classes), dtype=np.float64)
    for i in range(len(b_all)):
        s_all[i] = [True if v == 1 else False for v in b_all[i]]

    auroc, auprc, auroc_classes, auprc_classes = utils.compute_auc(t_all, p_all)
    f_measure, f1score_classes, precision, _, recall, _, specificity, _ = utils.compute_f_measure_and_precision_recall_specificity(
        t_all, b_all)
    kappa = utils.compute_kappa(t_all, b_all, num_classes=num_classes)
    hamming_loss = sklearn.metrics.hamming_loss(t_all, b_all)

print("####################################################################")
print(opt.pretrained_model_rhythm)
print(opt.pretrained_model_duration)
print(opt.pretrained_model_amplitude)
print(opt.pretrained_model_morphology)
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
