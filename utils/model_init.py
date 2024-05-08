from dataloader import *


def get_train_mode(opt):
    """Retrieve the specified training and validation methods based on the user's choice of loss type."""

    mode = opt.mode
    arch = opt.arch

    # Define dictionaries to map mode and architecture to their corresponding values
    mode_mapping = {
        'all': (
            'all_valid', opt.classes_all, opt.pretrained_model,
            dataloader_train),
        'rhythm': (
            'rhythm_valid', opt.classes_rhythm, opt.pretrained_model_rhythm,
            dataloader_train),
        'duration': (
            'duration_valid', opt.classes_duration, opt.pretrained_model_duration,
            dataloader_train),
        'amplitude': (
            'amplitude_valid', opt.classes_amplitude, opt.pretrained_model_amplitude,
            dataloader_train),
        'morphology': (
            'morphology_valid', opt.classes_morphology, opt.pretrained_model_morphology,
            dataloader_train)
    }

    arch_mapping = {
        'ResU_Dense': 'ResU_Dense',
        'ResU2D_LC': 'ResU2D_LC',
    }

    if mode not in mode_mapping:
        raise Exception("Loss type error!")

    if arch not in arch_mapping:
        raise Exception("Arch error!")

    validator_name, class_num, pretrained_model, dataloader = mode_mapping[mode]
    arch_name = arch_mapping[arch]

    print(f'You choose {mode} trainer ...')
    print(f'You choose {arch}...')

    # Import the specified architecture from models
    from models import ResU_Dense, ResU2D_LC

    model = ResU_Dense if arch_name == 'ResU_Dense' else ResU2D_LC
    model = model(nOUT=class_num)

    # Import the specified training and validation methods from trainer
    from .trainer import ms_train, ms_valid

    trainer = ms_train
    validator = ms_valid

    return model, trainer, validator, dataloader, pretrained_model


def get_valid_mode(opt):
    mode = opt.mode
    arch = opt.arch

    # Define dictionaries to map mode and architecture to their corresponding values
    mode_mapping = {
        'all': (
            'all_valid', opt.classes_all, opt.pretrained_model,
            dataloader_valid),
        'rhythm': (
            'rhythm_valid', opt.classes_rhythm, opt.pretrained_model_rhythm,
            dataloader_valid),
        'duration': (
            'duration_valid', opt.classes_duration, opt.pretrained_model_duration,
            dataloader_valid),
        'amplitude': (
            'amplitude_valid', opt.classes_amplitude, opt.pretrained_model_amplitude,
            dataloader_valid),
        'morphology': (
            'morphology_valid', opt.classes_morphology, opt.pretrained_model_morphology,
            dataloader_valid)
    }

    arch_mapping = {
        'ResU_Dense': 'ResU_Dense',
        'ResU2D_LC': 'ResU2D_LC',
    }

    if mode not in mode_mapping:
        raise Exception("Loss type error!")

    if arch not in arch_mapping:
        raise Exception("Arch error!")

    validator_name, class_num, pretrained_model, dataloader = mode_mapping[mode]
    arch_name = arch_mapping[arch]

    print(f'You choose {mode} trainer ...')
    print(f'You choose {arch}...')

    # Import the specified architecture from models
    from models import ResU_Dense, ResU2D_LC

    model = ResU_Dense if arch_name == 'ResU_Dense' else ResU2D_LC
    model = model(nOUT=class_num)

    return model, dataloader, pretrained_model, class_num


def get_inferece_mode(opt):
    arch = opt.arch

    arch_mapping = {
        'ResU_Dense': 'ResU_Dense',
        'ResU2D_LC': 'ResU2D_LC',
    }

    if arch not in arch_mapping:
        raise Exception("Arch error!")

    arch_name = arch_mapping[arch]

    print(f'You choose {arch}...')

    # Import the specified architecture from models
    from models import ResU_Dense, ResU2D_LC

    model = ResU_Dense if arch_name == 'ResU_Dense' else ResU2D_LC

    model_rhythm = model(nOUT=opt.classes_rhythm)
    model_duration = model(nOUT=opt.classes_duration)
    model_amplitude = model(nOUT=opt.classes_amplitude)
    model_morphology = model(nOUT=opt.classes_morphology)

    return [model_rhythm, model_duration, model_amplitude, model_morphology]


# get loss function
def get_loss(opt):
    """Retrieve the specified loss function based on user's choice."""

    loss_func = opt.loss_func
    DEVICE = opt.device
    print('You choose ' + loss_func + ' loss function ...')

    # Check if the chosen loss operation is 'base'
    if loss_func == 'BCE':
        import torch.nn as nn
        bce_loss = nn.BCEWithLogitsLoss().to(DEVICE)
        return bce_loss
    else:
        # If the loss operation type is not found, raise an exception
        raise Exception("Loss type error!")
