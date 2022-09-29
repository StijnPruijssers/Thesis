import wandb
from wandb.keras import WandbCallback
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import calc_metrics


class WandbClassificationCallback(WandbCallback):

    def __init__(self, monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=False, log_weights=False, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=[], data_type=None, predictions=1, generator=None,
                 input_type=None, output_type=None, log_evaluation=False,
                 validation_steps=None, class_colors=None, log_batch_frequency=None,
                 log_best_prefix="best_",
                 log_additional_train=False, log_additional_val=False, layers=1,
                 model_type='cnn'):

        super().__init__(monitor=monitor,
                         verbose=verbose,
                         mode=mode,
                         save_weights_only=save_weights_only,
                         log_weights=log_weights,
                         log_gradients=log_gradients,
                         save_model=save_model,
                         training_data=training_data,
                         validation_data=validation_data,
                         labels=labels,
                         data_type=data_type,
                         predictions=predictions,
                         generator=generator,
                         input_type=input_type,
                         output_type=output_type,
                         log_evaluation=log_evaluation,
                         validation_steps=validation_steps,
                         class_colors=class_colors,
                         log_batch_frequency=log_batch_frequency,
                         log_best_prefix=log_best_prefix,
                         log_additional_train=log_additional_train,
                         log_additional_val=log_additional_val,
                         layers=layers,
                         model_type=model_type)

        self.log_additional_train = log_additional_train
        self.log_additional_val = log_additional_val
        self.layers = layers
        self.model_type = model_type

    def on_epoch_end(self, epoch, logs={}):
        if self.generator:
            self.validation_data = next(self.generator)

        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        # Log additional metrics
        if self.log_additional_train:
            wandb.log(self._log_additional_metrics(data_split='train'), commit=False)

        if self.log_additional_val:
            wandb.log(self._log_additional_metrics(data_split='val'), commit=False)

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
                        epoch, self.monitor, self.best, self.current))
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _create_metric_dict(self, specificity, recall, precision, mcc, data_split='val', lay_no=0):
        metric_dict = {}
        if self.layers == 3:
            extension = '_lay{}'.format(lay_no + 1)
        else:
            extension = ''

        # Log overall metrics
        metric_dict['Mcc/{}_mcc{}'.format(data_split, extension)] = mcc
        metric_dict['Recall/{}_recall{}'.format(data_split, extension)] = np.mean(recall)
        metric_dict['Precision/{}_precision{}'.format(data_split, extension)] = np.mean(precision)
        metric_dict['Specificity/{}_specificity{}'.format(data_split, extension)] = np.mean(specificity)

        for class_no in range(specificity.shape[0]):
            metric_dict['Recall/{}_recall_{}{}'.format(data_split, self.labels[class_no], extension)] = \
                recall[class_no]
            metric_dict['Precision/{}_precision_{}{}'.format(data_split, self.labels[class_no], extension)] = \
                precision[class_no]
            metric_dict['Specificity/{}_specificity_{}{}'.format(data_split, self.labels[class_no], extension)] = \
                specificity[class_no]

        return metric_dict

    def _log_additional_metrics(self, data_split='val'):

        # Define data
        if data_split == 'val':
            x = self.validation_data[0]
            y = self.validation_data[1]

            # If it is a dict type, convert to numpy array
            if type(y) == dict:
                y = np.array([list(item) for item in y.values()])
                y = np.transpose(y, (1, 0, 2))

        elif data_split == 'train':
            x = self.training_data[0]
            y = self.training_data[1]

        # Predict data for calculating metrics
        predictions = self.model.predict(x, batch_size=y.shape[0])

        if self.layers == 3:
            mcc_mean = np.zeros(3)
            metric_dict = {}

            for lay_no in range(self.layers):
                # Calculate metrics
                specificity, recall, precision, mcc, confmat = calc_metrics(y_true=y[:, lay_no, :],
                                                                            y_pred=predictions[lay_no],
                                                                            labels=list(range(len(self.labels))))
                mcc_mean[lay_no] = mcc

                # Log metrics
                metric_dict_temp = self._create_metric_dict(specificity, recall, precision, mcc, data_split=data_split,
                                                            lay_no=lay_no)
                metric_dict = {**metric_dict, **metric_dict_temp}

            metric_dict['Mcc/{}_mcc_mean'.format(data_split)] = np.mean(mcc_mean)

        else:
            # Calculate metrics
            specificity, recall, precision, mcc, confmat = calc_metrics(y_true=y, y_pred=predictions,
                                                                        labels=list(range(len(self.labels))))

            # Log metrics
            metric_dict = self._create_metric_dict(specificity, recall, precision, mcc, data_split=data_split)

        return metric_dict
