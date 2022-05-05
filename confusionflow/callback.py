import json, os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from utils import create_dir

class Logs(tf.keras.callbacks.Callback):
    
    # Pass in validation data & loss function
    def __init__(self, data, loss=None):
        self.data = data
        self.loss = loss
        self.logs = {'accuracy': []}
    
    def on_epoch_begin(self, epoch, logs=None):
        self.log_output(epoch)
    
    # Save test performance for each epoch
    def log_output(self, epoch):
        """
        Record model confusion and model accuracy after each epoch
        Input: epoch number
        """
        feats, labels = self.data
        preds = np.argmax(self.model.predict(feats, verbose=0), axis=1)
        
        if self.loss == 'categorical_crossentropy':
            labels = np.argmax(labels, axis=1)
        elif self.loss == 'sparse_categorical_crossentropy':
            pass # Don't change labels
        else:
            raise ValueError("loss `{}` is not supported".format(self.loss))
        
        cm = tf.math.confusion_matrix(labels, preds).numpy().tolist()
        self.logs[epoch] = cm
        self.logs['accuracy'].append(accuracy_score(labels, preds))
    
    def export(self, log_dir, model_id):
        """
        Export logs to log file
        Input: directory name, model name
        """
        create_dir(log_dir)
        log_path = os.path.join(log_dir, model_id + '.json')
        with open(log_path, 'w') as log_file:
            json.dump(self.logs, log_file)