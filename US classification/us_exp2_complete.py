from models_us import get_us_cnn_complete_simple
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
from split_preprocess_augment import *
from sklearn.metrics import confusion_matrix
import copy
from utils import maybe_mkdir_p, calc_metrics
from data_generator import DataGenerator

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# The GPU id to use, usually either 0,1,2 or 3.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_START_METHOD'] = 'spawn'

'''
DEFINE PARAMS
'''

run_params = {'dataset_name': 'complete_us',
              'hyper_param_name': 'US_complete_dryrun',
              'max_trials': 1000,
              'exec_per_trial': 1,
              'overwrite': False,
              'class_weighting': True,
              'n_folds': 5,
              'epochs': 2,
              'random_state': 3,
              'save_model': True,
              'normalization': 'minmax',
              'wandb_mode': 'online',
              'label_names': ['Healthy', 'Tumor']}

maybe_mkdir_p('wandb_data/'+run_params['hyper_param_name'])

if run_params['save_model']:
    maybe_mkdir_p('Models/' + run_params['hyper_param_name'])

# Load data
with open('Data processed/complete_us_cropped2.pkl', 'rb') as file:
    data = pickle.load(file)

# Get class weights for training set for relative labels labels
if run_params['class_weighting']:
    data['class_weights'] = \
        compute_class_weight('balanced', classes=np.unique(data['train_labels']), y=data['train_labels'])
else:
    data['class_weights'] = np.ones(len(run_params['label_names']))


class myHypermodel(kt.HyperModel):
    def __init__(self, name=None, tunable=True, class_weights=[1, 1], x_size=(230, 344)):
        self.name = name
        self.tunable = tunable
        self.class_weights = class_weights
        self.x_size = x_size

        self._build = self.build
        self.build = self._build_wrapper

    def build(self, hp):
        # Create hypermodel for hyperparameter optimalisation.
        depth = hp.Fixed('depth', value=5)
        dropout_rate = hp.Float('dropout', min_value=0.3, max_value=0.5, step=0.1)
        filters = hp.Choice('filters', [2, 4, 8, 16, 32, 64])
        lr = hp.Float('lr', min_value=0.001, max_value=0.01, step=0.002)
        kernel_size_1 = hp.Fixed('kernel_size_1', value=3)
        kernel_size_2 = hp.Fixed('kernel_size_2', value=3)
        opt = hp.Choice('opt', ['adam', 'sgdm'])

        model = get_us_cnn_complete_simple(input_shape=(self.x_size[0], self.x_size[1], 1),
                                           dropout_rate=dropout_rate,
                                           depth=depth,
                                           lr=lr,
                                           filters=filters,
                                           kernel_size=(kernel_size_1, kernel_size_2),
                                           strides=(1, 1),
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
        batch_size = trial.hyperparameters.Choice('batch_size', [16])

        # Set up stratified cross validation
        skf = StratifiedKFold(n_splits=params['n_folds'], random_state=params['random_state'], shuffle=True)
        val_loss = np.zeros((params['n_folds']))
        mcc_tot = np.empty(params['n_folds'])

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

            training_generator = DataGenerator(train_idx, data['train_labels'],
                                               batch_size=trial.hyperparameters.values['batch_size'],
                                               dim=(230, 344), shuffle=True)

            validation_generator = DataGenerator(val_idx, data['train_labels'],
                                                 batch_size=trial.hyperparameters.values['batch_size'],
                                                 dim=(230, 344), shuffle=True)
            # Select data
            train_us, val_us = data['train_us'][train_idx], data['train_us'][val_idx]
            train_labels_fold, val_labels_fold = data['train_labels'][train_idx], data['train_labels'][val_idx]
            train_labels_fold = train_labels_fold.astype('float32')
            val_labels_fold = val_labels_fold.astype('float32')

            train_labels_fold = to_categorical(train_labels_fold, num_classes=2)
            val_labels_fold = to_categorical(val_labels_fold, num_classes=2)

            # Prepare data for CNN
            x_train = np.expand_dims(train_us, axis=-1)
            x_val = np.expand_dims(val_us, axis=-1)

            # Build model and clear for next run
            model = None  # Clear model before next run
            model_best = None
            model = self.hypermodel.build(trial.hyperparameters)

            # Fit model
            wandb.setup(wandb.Settings(
                program='Z:/Personal folder/Stijn Pruijssers (M3)_UT_2021/Software/Code/CRNN_git3/us_exp2_complete.py',
                program_relpath='us_exp2_complete.py', mode=params['wandb_mode']))

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
                                            layers=1))

            history = model.fit(training_generator,
                                validation_data=(x_val, val_labels_fold), *fit_args, **copied_fit_kwargs)

            val_loss[i] = np.min(history.history['val_loss'])

            # Save model at end of training
            if params['save_model']:
                model.save(run_dir + '/model_end.h5')
            else:
                print('No model saved at and of training')

            # Load best model
            if params['save_model']:
                model_best = load_model(run_dir + '/model_end.h5', compile=False)
            else:
                model_best = load_model('model_temp_test.h5', compile=False)

            if trial.hyperparameters.values['opt'] == 'adam':
                opt = Adam(learning_rate=trial.hyperparameters.values['lr'])
            elif trial.hyperparameters.values['opt'] == 'sgdm':
                opt = SGD(learning_rate=trial.hyperparameters.values['lr'])
            else:
                raise NotImplementedError('No valid optimizer is defined, available options are adam or sgdm')

            loss = categorical_focal_loss(alpha=data['class_weights'], gamma=0)
            model_best.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

            # Predict validation set
            predictions = model_best.predict(x_val, batch_size=x_val.shape[0])

            # Calculate and log metrics
            confmat = confusion_matrix(np.argmax(val_labels_fold, axis=1),
                                       np.argmax(predictions, axis=1), labels=list(range(len(params['label_names']))))

            fig, ax = plt.subplots()
            df_cm = pd.DataFrame(confmat, index=params['label_names'],
                                 columns=params['label_names'])
            sn.heatmap(df_cm, annot=True)
            ax.set_xlabel('Pred', fontweight='bold')
            ax.set_ylabel('True', fontweight='bold')
            fig.savefig("Temp/tempconf.png")
            conf_image = wandb.Image("Temp/tempconf.png")
            wandb.log({'confmat': conf_image})
            plt.close(fig)

            # Calculate metrics
            specificity, recall, precision, mcc = calc_metrics(confusion_matrix=confmat,
                                                               y_true=val_labels_fold,
                                                               y_pred=predictions)

            # Log summarizing metrics in table

            # Log all average metrics
            wandb.run.summary['Mcc'] = mcc
            wandb.run.summary['Recall'] = np.mean(recall)
            wandb.run.summary['Precision'] = np.mean(precision)
            wandb.run.summary['Specificity'] = np.mean(specificity)

            # Log per class metrics
            for class_no, class_name in enumerate(params['label_names']):
                wandb.run.summary['Recall_{}'.format(class_name)] = recall[class_no]
                wandb.run.summary['Precision{}'.format(class_name)] = precision[class_no]
                wandb.run.summary['Specificity{}'.format(class_name)] = specificity[class_no]

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
    oracle=kt.oracles.BayesianOptimization(objective=kt.Objective(name='val_mcc', direction='max'),
                                           max_trials=run_params['max_trials']))

# Create checkpoints
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=25)

# Create search
tuner.search_space_summary()
tuner.search(verbose=1, callbacks=[es], shuffle=True, data=data, params=run_params, epochs=run_params['epochs'])
