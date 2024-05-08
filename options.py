class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # Global settings
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Size of each batch during training.')
        parser.add_argument('--nepoch', type=int, default=50,
                            help='Total number of training epochs.')
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='Type of optimizer to use during training (e.g., "adam", "sgd").')
        parser.add_argument('--lr_initial', type=float, default=1e-3,
                            help='Initial learning rate.')
        parser.add_argument("--decay_epoch", type=int, default=20,
                            help="Epoch number from which to start learning rate decay.")
        parser.add_argument('--valid_ratio', type=float, default=0.2,
                            help='Proportion of data to be used for validation.')

        # Training settings
        parser.add_argument('--arch', type=str, default='ResU_Dense',
                            help='Model architecture to be used for training (e.g., "ResU_Dense", "ResU2D_LC").')
        parser.add_argument('--mode', type=str, default='all',
                            help='Train mode (e.g., "all", "rhythm", "duration", "amplitude", "morphology".')
        parser.add_argument('--loss_func', type=str, default='BCE',
                            help='loss function for training (e.g., "BCE").')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to be used for training ("cuda" for GPU, "cpu" for CPU).')
        parser.add_argument('--unlabels', type=str, default='./data_info/target/remove_labels.npy',
                            help='List of unclassified labels in the database')
        parser.add_argument('--mapping_info', type=str, default='./data_info/target/mapping_labels.npy',
                            help='Label Integration Information')
        parser.add_argument('--target_all', type=str, default='./data_info/target/target_labels.npy',
                            help='List of 20 labels targeted for classification')
        parser.add_argument('--target_rhythm', type=str, default='./data_info/target/target_labels_rhythm.npy',
                            help='List of 5 Rhythm classification targets')
        parser.add_argument('--target_duration', type=str, default='./data_info/target/target_labels_duration.npy',
                            help='List of 5 duration classification targets')
        parser.add_argument('--target_amplitude', type=str, default='./data_info/target/target_labels_amplitude.npy',
                            help='List of 6 amplitude classification targets')
        parser.add_argument('--target_morphology', type=str, default='./data_info/target/target_labels_morphology.npy',
                            help='List of 4 morphology classification targets')

        # Network settings
        parser.add_argument('--classes_all', type=int, default=20,
                            help='Number of final output classes')
        parser.add_argument('--classes_rhythm', type=int, default=5,  # Rhythm diagnosis (AFIB, CAVB, GSVT, SB, SR)
                            help='Number of final output classes from the multi-label classification model')
        parser.add_argument('--classes_duration', type=int, default=5,
                            # Duration diagnosis (1AVB, APB, CLBBB, CRBBB, VPB)
                            help='Number of final output classes from the multi-label classification model')
        parser.add_argument('--classes_amplitude', type=int, default=6,
                            # Amplitude diagnosis (LAD, LAFB, LQV, LVH, RAD, RVH)
                            help='Number of final output classes from the multi-label classification model')
        parser.add_argument('--classes_morphology', type=int, default=4,  # Morphology diagnosis (PACE, QWA, STTA, WPW)
                            help='Number of final output classes from the multi-label classification model')
        parser.add_argument('--leads', type=int, default=12,
                            help='Number of ECG leads to be considered.')

        # Pretrained model settings
        parser.add_argument('--log_name', type=str, default='240403',
                            help='Name used for logging purposes.')
        parser.add_argument('--pretrained', type=bool, default=False,
                            help='True if loading weights from a pretrained model')
        parser.add_argument('--pretrained_model', type=str,
                            default='./log/ResU_Dense_all_240403/models/chkpt_56.pt',
                            help='Path to the pretrained model weights.')
        parser.add_argument('--pretrained_model_rhythm', type=str,
                            default='./log/ResU_Dense_rhythm_240403/models/chkpt_43.pt',
                            help='Path to the pretrained model weights.')
        parser.add_argument('--pretrained_model_duration', type=str,
                            default='./log/ResU_Dense_duration_240403/models/chkpt_36.pt',
                            help='Path to the pretrained model weights.')
        parser.add_argument('--pretrained_model_amplitude', type=str,
                            default='./log/ResU_Dense_amplitude_240403/models/chkpt_59.pt',
                            help='Path to the pretrained model weights.')
        parser.add_argument('--pretrained_model_morphology', type=str,
                            default='./log/ResU_Dense_morphology_240403/models/chkpt_49.pt',
                            help='Path to the pretrained model weights.')

        # Dataset settings
        parser.add_argument('--fs', type=int, default=500,
                            help='Sampling frequency of the ECG data.')
        parser.add_argument('--data_length', type=int, default=10,
                            help='data length (sec) of the ECG data.')
        parser.add_argument('--samples', type=int, default=4096,
                            help='Number of samples to consider from the ECG data.')
        parser.add_argument('--dirs_for_train', type=str,
                            default='../../workspace_ecg/Dataset/physionet_large_scale/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords',
                            help='Directory path where ECG dataset for training and validation is located.')

        return parser
