
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os
from flytekit import task, workflow, Resources
import typing
from flytekit.types.file import PythonPickledFile

@dataclass_json
@dataclass
class Hyperparameters(object):
    learning_rate: float = 0.1
    batch_size: int = 10
    epochs: int = 100
    log_interval: int = 20


def loadData():
    train_data = load_iris()
    data = train_data['data']
    labels = train_data['target'].reshape(-1,1)
    total_data = np.hstack((data,labels))
    np.random.shuffle(total_data)
    train_length = int(len(total_data) * 0.8)
    train = total_data[0:train_length, :-1]
    test = total_data[train_length:, :-1]
    train_label = total_data[0:train_length, -1].reshape(-1, 1)
    test_label = total_data[train_length:, -1].reshape(-1, 1)
    print(data.shape, labels.shape, train.shape, test.shape, train_label.shape, test_label.shape)
    return data, labels, train, test, train_label, test_label


if os.getenv("SANDBOX") != "":
    print(f"SANDBOX ENV: '{os.getenv('SANDBOX')}'")

    mem = "100Mi"
    gpu = "0"
    storage = "500Mi"
else:
    print(f"SANDBOX ENV: '{os.getenv('SANDBOX')}'")

    mem = "3Gi"
    gpu = "0"
    storage = "1Gi"


TrainingOutputs = typing.NamedTuple(
    "TrainingOutputs",
    train_losses=typing.List[float],
    train_accuracies=typing.List[float],
    all_acc=float,
    test_acc=float,
)

@task(
    retries=2,
    cache=False,
    cache_version="1.0",
    requests=Resources(gpu=gpu, mem=mem, storage=storage),
    limits=Resources(gpu=gpu, mem=mem, storage=storage),
)
def tensorflowIrisTask(hp: Hyperparameters) -> TrainingOutputs:
    import tensorflow as tf
    
    data, labels, train, test, train_label, test_label = loadData()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu',input_shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train, train_label))
    train_dataset = train_dataset.batch(hp.batch_size)
    

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=hp.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
 
    history = model.fit(train_dataset, epochs=hp.epochs)
    
    test_loss, test_acc = model.evaluate(test, test_label)
    all_loss, all_acc = model.evaluate(data, labels)
    print(history.history.keys())
    
    training_accuracies = history.history['sparse_categorical_accuracy']
    training_losses = history.history['loss']
    
    return TrainingOutputs(train_losses=training_losses,
        train_accuracies=training_accuracies,
        all_acc=all_acc,
        test_acc=test_acc,
    )
    
    
@workflow
def tensorflowTrainingWorkflow(
    hp: Hyperparameters = Hyperparameters(epochs=100, batch_size=10)
) -> TrainingOutputs:
    return tensorflowIrisTask(hp=hp)


if __name__=='__main__' :
    
    train_losses, test_losses, all_acc, test_acc = tensorflowTrainingWorkflow(hp=Hyperparameters(epochs=100, batch_size=10, learning_rate=0.01))
    
    print(train_losses, test_losses, all_acc, test_acc)