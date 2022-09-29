from models_drs import get_drs_rnn_3layer
import os
import kerastuner as kt
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
from kerastuner.engine import tuner_utils
from sklearn.utils.class_weight import compute_class_weight
import wandb
from custom_callbacks import WandbClassificationCallback
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from kerastuner import HyperParameters
from custom_losses import categorical_focal_loss
from tensorflow.keras.optimizers import Adam, SGD
from utils import remove_nans
from split_preprocess_augment import *
from sklearn.metrics import confusion_matrix
import copy
from utils import maybe_mkdir_p, calc_metrics

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# The GPU id to use, usually either 0,1,2 or 3.
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['WANDB_START_METHOD'] = 'spawn'

'''
DEFINE PARAMS
'''

run_params = {'dataset_name': 'drs_checked_hard',
              'hyper_param_name': 'DRS_Exp1_RNN',
              'max_trials': 1000,
              'exec_per_trial': 1,
              'overwrite': False,
              'class_weighting': True,
              'n_folds': 4,
              'epochs': 100,
              'random_state': 3,
              'save_model': True,
              'normalization': 'minmax',
              'wandb_mode': 'online',
              'label_names': ['Fat', 'Healthy', 'Tumor', 'Fibrosis']}

maybe_mkdir_p('wandb_data/'+run_params['hyper_param_name'])

if run_params['save_model']:
    maybe_mkdir_p('Models/' + run_params['hyper_param_name'])

# Load dataset
with open('Data processed/' + run_params['dataset_name'] + '.pkl', 'rb') as file:
    data = pickle.load(file)

# Normalize dataset as indicated
if run_params['normalization'] == 'magnitude':
    data['train_fib'], data['val_fib'], data['test_fib'] = \
        magnitude_norm(data['train_fib']), magnitude_norm(data['val_fib']), magnitude_norm(data['test_fib'])
elif run_params['normalization'] == 'snv':
    data['train_fib'], data['val_fib'], data['test_fib'] = \
        snv(data['train_fib']), snv(data['val_fib']), snv(data['test_fib'])
elif run_params['normalization'] == '800nm':
    data['train_fib'], data['val_fib'], data['test_fib'] = \
        norm_800(data['train_fib']), norm_800(data['val_fib']), norm_800(data['test_fib'])
elif run_params['normalization'] == 'minmax':
    data['train_fib'], data['val_fib'], data['test_fib'] = \
        drs_min_max_norm(data['train_fib']), drs_min_max_norm(data['val_fib']), drs_min_max_norm(data['test_fib'])

'''
FORMAT DATA FOR 3LAYER EXPERIMENT
'''
# Get labels of all 3 layers and remove NaN values for train and test set
train_meta = data['train_meta']
train_meta = train_meta.loc[data['train_sample_ids'].astype('int64')]
train_labels = train_meta[['PA_LabelLayer1', 'PA_LabelLayer2', 'PA_LabelLayer3']].to_numpy()
train_labels = remove_nans(train_labels)
data['train_labels'] = train_labels

test_meta = data['test_meta']
test_meta = test_meta.loc[data['test_sample_ids'].astype('int64')]
test_labels = test_meta[['PA_LabelLayer1', 'PA_LabelLayer2', 'PA_LabelLayer3']].to_numpy()
test_labels = remove_nans(test_labels)
data['test_labels'] = test_labels

# Create stratification labels based on first two layer combinations
labels_stratify = data['train_meta'][['PA_LabelLayer1', 'PA_LabelLayer2', 'PA_LabelLayer3']].to_numpy()
labels_stratify = remove_nans(labels_stratify)
pat_ids = data['train_meta']['PatientID'].to_numpy()
labels_stratify, pat_stratify, df_combs_final_max = get_combination_stratify_labels(labels_stratify, pat_ids,
                                                                                    k_fold=True, simple=True)

labels_stratify[labels_stratify == 7] = 5  # Set fibrosis samples under 1 sample to enable k-fold cross validation
labels_stratify[labels_stratify > 7] = 7
data['labels_stratify'] = labels_stratify  # Put stratification labels in data dict

# Select wavelengths from 600 to 1600
data['train_fib'] = data['train_fib'][:, 200::, :]
data['val_fib'] = data['val_fib'][:, 200::, :]
data['test_fib'] = data['test_fib'][:, 200::, :]


class myHypermodel(kt.HyperModel):
    def __init__(self, name=None, tunable=True, class_weights=[1, 1, 1, 1], x_size=1001, loss_weights=[1, 1, 1]):
        self.name = name
        self.tunable = tunable
        self.class_weights = class_weights
        self.x_size = x_size
        self.loss_weights = loss_weights
        self._build = self.build
        self.build = self._build_wrapper

    def build(self, hp):
        depth = hp.Int('depth', min_value=1, max_value=3, step=1)
        dropout_rate = hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)
        hidden_units = hp.Choice('hidden_units', [16, 32, 64, 128])
        lr = hp.Choice('lr', [0.0001, 0.0005, 0.001, 0.005, 0.01])
        rrn_type = hp.Choice('rnn_type', ['GRU', 'LSTM'])
        opt = hp.Choice('opt', ['adam', 'sgdm'])
        bidirectional = hp.Boolean('bidirectional', default=True)
        gamma = hp.Int('gamma', min_value=0, max_value=5, step=1)
        model = get_drs_rnn_3layer(input_shape=(6, self.x_size),
                                   depth=depth,
                                   dropout_rate=dropout_rate,
                                   hidden_units=hidden_units,
                                   lr=lr, rnn_type=rrn_type,
                                   bidirectional=bidirectional,
                                   gamma=gamma,
                                   class_weights=self.class_weights,
                                   loss_weights=self.loss_weights,
                                   optimizer=opt)

        return model


# Create subclass of tuner base class and write custom training loop
class MyTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, *fit_args, **fit_kwargs):

        # Handle any callbacks passed to `fit`.
        copied_fit_kwargs = copy.copy(fit_kwargs)
        callbacks = fit_kwargs.pop('callbacks', [])
        callbacks = self._deepcopy_callbacks(callbacks)
        self._configure_tensorboard_dir(callbacks, trial.trial_id)

        # `TunerCallback` calls:
        # - `Tuner.on_epoch_begin`
        # - `Tuner.on_batch_begin`
        # - `Tuner.on_batch_end`
        # - `Tuner.on_epoch_end`
        # These methods report results to the `Oracle` and save the trained Model. If
        # you are subclassing `Tuner` to write a custom training loop, you should
        # make calls to these methods within `run_trial`.

        callbacks.append(tuner_utils.TunerCallback(self, trial))  # Add keras tuner callback

        copied_fit_kwargs.pop('data')
        copied_fit_kwargs.pop('params')

        # Extract data and params from fit_kwargs
        # (dirty way, else MultiExecution tuner class would need to be subclassed as well)
        data = fit_kwargs['data']
        params = fit_kwargs['params']

        # Make model for saving if needed

        if params['save_model']:
            trial_dir = 'Models/' + params['hyper_param_name'] + '/' + trial.trial_id
            maybe_mkdir_p(trial_dir)

        # Set batch size as hyperparameter
        copied_fit_kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [8, 16, 32, 64, 128])

        # Set up stratified cross validation
        skf = StratifiedKFold(n_splits=params['n_folds'], random_state=params['random_state'], shuffle=True)
        val_loss = np.zeros((params['n_folds']))

        for i, (train_fold, val_fold) in enumerate(skf.split(data['train_ids'], data['labels_stratify'])):
            print("Running Fold", i + 1, "/", params['n_folds'])

            # Get selected indices of patient IDs
            train_idx = np.array([])
            for item in data['train_ids'][train_fold]:
                train_idx = np.append(train_idx, np.argwhere(data['train_ids_augment'] == item))
            train_idx = train_idx.astype('int64')

            val_idx = np.array([])
            for item in data['train_ids'][val_fold]:
                val_idx = np.append(val_idx, np.argwhere(data['train_ids_augment'] == item))
            val_idx = val_idx.astype('int64')

            # Select data
            train_fib, val_fib = data['train_fib'][train_idx], data['train_fib'][val_idx]
            train_labels_fold, val_labels_fold = data['train_labels'][train_idx], data['train_labels'][val_idx]
            train_labels_fold = train_labels_fold.astype('float32')
            val_labels_fold = val_labels_fold.astype('float32')

            # Calculate label-weights over complete dataset so loss values are comparable between folds
            if params['class_weighting']:

                weights_lay1 = compute_class_weight('balanced', classes=np.unique(data['train_labels'][:, 0]),
                                                    y=data['train_labels'][:, 0])
                weights_lay2 = compute_class_weight('balanced', classes=np.unique(data['train_labels'][:, 1]),
                                                    y=data['train_labels'][:, 1])
                weights_lay3 = compute_class_weight('balanced', classes=np.unique(data['train_labels'][:, 2]),
                                                    y=data['train_labels'][:, 2])

                weights = np.vstack((weights_lay1, weights_lay2, weights_lay3))
            else:
                weights = np.ones((3, len(params['label_names'])))

            # Turn to one-hot encoding
            train_labels_fold = to_categorical(train_labels_fold, num_classes=len(params['label_names']))
            val_labels_fold = to_categorical(val_labels_fold, num_classes=len(params['label_names']))

            # Downsample spectra as indicated
            down_rate = trial.hyperparameters.Choice('down_rate', [200, 300, 500, 1001])
            train_fib = interpolate_array1d(train_fib, down_rate)
            val_fib = interpolate_array1d(val_fib, down_rate)

            # Prepare data for RNN
            x_train = np.transpose(train_fib, (0, 2, 1))
            x_val = np.transpose(val_fib, (0, 2, 1))

            # Set weights and size for model creation
            self.hypermodel.hypermodel.class_weights = weights
            self.hypermodel.hypermodel.x_size = x_train.shape[-1]

            # Set loss_weights if necessary
            weighted_losses = trial.hyperparameters.Boolean('weighted_losses', default=False)
            if weighted_losses:
                self.hypermodel.hypermodel.loss_weights = list(data['loss_weights'])
            else:
                self.hypermodel.hypermodel.loss_weights = [1, 1, 1]

            # Build model and clear for next run
            model = None  # Clear model before next run
            model_best = None
            model = self.hypermodel.build(trial.hyperparameters)

            # Fit model
            wandb.setup(wandb.Settings(
                program='Z:/Personal folder/Stijn Pruijssers (M3)_UT_2021/Software/Code/CRNN_git3/drs_exp1_3layer_rnn.py',
                program_relpath='drs_exp1_3layer_rnn.py', mode=params['wandb_mode']))

            run = wandb.init(name='fold_{}'.format(i), project=self.project_name, reinit=True, group=trial.trial_id,
                             config=trial.hyperparameters.values, dir='wandb_data/'+params['hyper_param_name'])

            # Copy callbacks and add WandB callback after initializing runs for every fold
            copied_fit_kwargs['callbacks'] = copy.deepcopy(callbacks)

            if params['save_model']:
                run_dir = trial_dir + '/' + 'fold_{}'.format(i)
                maybe_mkdir_p(run_dir)
                mc = ModelCheckpoint(run_dir + '/model_best.h5', monitor='val_loss', verbose=1, save_best_only=True)
                copied_fit_kwargs['callbacks'].append(mc)
            else:
                mc = ModelCheckpoint('model_temp_test.h5', monitor='val_loss', verbose=1, save_best_only=True)
                copied_fit_kwargs['callbacks'].append(mc)

            copied_fit_kwargs['callbacks'].append(
                WandbClassificationCallback(training_data=(x_train, train_labels_fold),
                                            validation_data=(x_val, val_labels_fold),
                                            save_model=False,
                                            labels=params['label_names'],
                                            log_additional_train=True,
                                            log_additional_val=True,
                                            layers=3))

            history = model.fit(x=x_train,
                                y={"OutputLayer1": train_labels_fold[:, 0, :],
                                   "OutputLayer2": train_labels_fold[:, 1, :],
                                   "OutputLayer3": train_labels_fold[:, 2, :]},
                                validation_data=(x_val,
                                                 {"OutputLayer1": val_labels_fold[:, 0, :],
                                                  "OutputLayer2": val_labels_fold[:, 1, :],
                                                  "OutputLayer3": val_labels_fold[:, 2, :]}), *fit_args,
                                **copied_fit_kwargs)
            val_loss[i] = np.min(history.history['val_loss'])

            # Save model at end of training
            if params['save_model']:
                model.save(run_dir + '/model_end.h5')
            else:
                print('No model saved at and of training')

            # Load best model
            if params['save_model']:
                model_best = load_model(run_dir + '/model_best.h5', compile=False)
            else:
                model_best = load_model('model_temp_test.h5', compile=False)

            if trial.hyperparameters.values['opt'] == 'adam':
                opt = Adam(learning_rate=trial.hyperparameters.values['lr'])
            elif trial.hyperparameters.values['opt'] == 'sgdm':
                opt = SGD(learning_rate=trial.hyperparameters.values['lr'])
            else:
                raise NotImplementedError('No valid optimizer is defined, available options are adam or sgdm')

            loss = {
                'OutputLayer1': categorical_focal_loss(alpha=[weights[0]], gamma=trial.hyperparameters.values['gamma']),
                'OutputLayer2': categorical_focal_loss(alpha=[weights[1]], gamma=trial.hyperparameters.values['gamma']),
                'OutputLayer3': categorical_focal_loss(alpha=[weights[2]], gamma=trial.hyperparameters.values['gamma'])}

            model.compile(loss=loss, optimizer=opt, metrics=['accuracy'], loss_weights=list(data['loss_weights']))

            # Predict validation set
            predictions = model_best.predict(x_val, batch_size=x_val.shape[0])
            mcc_mean = np.array([0, 0, 0])

            for lay_no in range(3):  # For every layer

                # Calculate overall confusion matrix
                confmat = confusion_matrix(np.argmax(val_labels_fold[:, lay_no, :], axis=1),
                                           np.argmax(predictions[lay_no], axis=1),
                                           labels=list(range(len(params['label_names']))))

                # Visualize and log confusion matrix
                fig, ax = plt.subplots()
                df_cm = pd.DataFrame(confmat, index=params['label_names'],
                                     columns=params['label_names'])
                sn.heatmap(df_cm, annot=True)
                ax.set_xlabel('Pred', fontweight='bold')
                ax.set_ylabel('True', fontweight='bold')
                fig.savefig("Temp/tempconf.png")
                conf_image = wandb.Image("Temp/tempconf.png")
                wandb.log({'confmat_{}'.format(lay_no + 1): conf_image})
                plt.close(fig)

                # Calculate metrics
                specificity, recall, precision, mcc = calc_metrics(confusion_matrix=confmat,
                                                                   y_true=val_labels_fold[:, lay_no, :],
                                                                   y_pred=predictions[lay_no])
                mcc_mean[lay_no] = mcc  # Save layer mcc

                # Log summarizing metrics in table
                # Log all average metrics
                wandb.run.summary['Mcc_{}'.format(lay_no + 1)] = mcc
                wandb.run.summary['Recall_{}'.format(lay_no + 1)] = np.mean(recall)
                wandb.run.summary['Precision_{}'.format(lay_no + 1)] = np.mean(precision)
                wandb.run.summary['Specificity_{}'.format(lay_no + 1)] = np.mean(specificity)

                # Log per class metrics
                for class_no, class_name in enumerate(params['label_names']):
                    wandb.run.summary['Recall_{}_lay{}'.format(class_name, lay_no + 1)] = recall[class_no]
                    wandb.run.summary['Precision{}_lay{}'.format(class_name, lay_no + 1)] = precision[class_no]
                    wandb.run.summary['Specificity{}_lay{}'.format(class_name, lay_no + 1)] = specificity[class_no]
            wandb.run.summary['Mean_mcc'] = np.mean(mcc_mean)
            run.finish()

        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_loss)})

    def on_epoch_end(self, trial, model, epoch, logs=None):
        """A hook called at the end of every epoch.

        # Arguments:
            trial: A `Trial` instance.
            model: A Keras `Model`.
            epoch: The current epoch number.
            logs: Dict. Metrics for this epoch. This should include
              the value of the objective for this epoch.
        """


# Create tuner with subclassed tuner
save_dir = os.path.normpath('C:/Users/s.pruijssers/KerasTuner')

tuner = MyTuner(
    hypermodel=myHypermodel(),
    overwrite=run_params['overwrite'],
    directory=save_dir,
    project_name=run_params['hyper_param_name'],
    oracle=kt.oracles.BayesianOptimization(objective=kt.Objective(name='val_loss', direction='min'),
                                           max_trials=run_params['max_trials']))

# Create checkpoints
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=50)

# Create search
tuner.search_space_summary()
tuner.search(verbose=1, callbacks=[es], shuffle=True, data=data, params=run_params, epochs=run_params['epochs'])
