import os
import argparse
import torch
import options
import utils
import datetime
import random
import numpy as np
import time

######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
# Parse command-line arguments and configurations for the training experiment
opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification')).parse_args()
print(opt)

######################################################################################################################
#                                    Set a model (check point) and a log folder                                      #
######################################################################################################################
# Get the absolute path of the current script and initialize directories for logging and model storage
dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
print(dir_name)

log_dir = os.path.join(dir_name, 'log', opt.arch + '_' + opt.mode + '_' + opt.log_name)
utils.mkdir(log_dir)
print("Now time is : ", datetime.datetime.now().isoformat())
tboard_dir = os.path.join(log_dir, 'logs')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(model_dir)  # make a dir if there is no dir (given path)
utils.mkdir(tboard_dir)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# Set up the computation device (either CPU or GPU) and initialize random seeds for reproducibility
DEVICE = torch.device(opt.device)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

model, trainer, validator, dataloader, pretrained_model = utils.get_train_mode(opt)
loss_calculator = utils.get_loss(opt)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_initial)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch, gamma=0.1)

total_params = utils.cal_total_params(model)
print('total params (gen)  : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# If a pre-trained model is provided, load its parameters
epoch_start_idx = 1
if opt.pretrained:
    print('Load the pretrained model...')
    chkpt = torch.load(pretrained_model)
    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])
    epoch_start_idx = chkpt['epoch'] + 1
    print('Resuming Start Epoch: ', epoch_start_idx)
    utils.optimizer_to(optimizer, DEVICE)

######################################################################################################################
#                                               Create Dataloader                                                    #
######################################################################################################################
# Create dataloaders for training and validation
train_loader, valid_loader = dataloader(opt)
print("Sizeof training set: ", train_loader.__len__(),
      ", sizeof validation set: ", valid_loader.__len__())
model = model.to(DEVICE)

######################################################################################################################
#                                             Main program - train                                                   #
######################################################################################################################
# Initialize tensorboard writer and training log file
writer = utils.Writer(tboard_dir)
train_log_fp = open(model_dir + '/train_log.txt', 'a')
max_epoch = 0
max_auprc = 0.0

print('Train start...')
# Main training loop
for epoch in range(epoch_start_idx, opt.nepoch + 1):
    st_time = time.time()

    # Train
    train_loss = trainer(model, train_loader, loss_calculator, optimizer,
                         writer, epoch, DEVICE)

    # Save the model's state after each epoch
    save_path = str(model_dir + '/chkpt_%d.pt' % epoch)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

    # Update the learning rate based on the scheduler
    scheduler.step()

    valid_loss, auprc, f1 = validator(model, valid_loader, loss_calculator, writer, epoch, DEVICE)
    print('AUPRC {:.6f} F1 {:.6f}'.format(auprc, f1))
    train_log_fp.write('AUPRC {:.6f} F1 {:.6f}'.format(auprc, f1))

    # Print the training and validation loss and AUPRC for the current epoch
    print('EPOCH[{}] T {:.6f} | V {:.6f}  takes {:.3f} seconds'
          .format(epoch, train_loss, valid_loss, time.time() - st_time))

    # Write results to the training log
    train_log_fp.write('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds'
                       .format(epoch, train_loss, valid_loss, time.time() - st_time))

print('Training has been finished.')
train_log_fp.close()
