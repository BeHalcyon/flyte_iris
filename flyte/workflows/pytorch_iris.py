
from  sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os
from flytekit import task, workflow, Resources
import typing
from flytekit.types.file import PythonPickledFile
import torch
import torch.nn as nn
import torch.utils.data as Data

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


# 网络类
class Model(nn.Module):
    def __init__(self):
        import torch
        
        super(Model, self).__init__()
        self.fc=nn.Sequential( # 添加神经元以及激活函数
            nn.Linear(4, 30),
            nn.ReLU(),
            # nn.Linear(20, 30),
            # nn.ReLU(),
            nn.Linear(30, 3)
        )
        self.mse = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.parameters(),lr=0.1)
        self.loss = None
        
    def forward(self, inputs):
        outputs=self.fc(inputs)
        return outputs
    
    def train(self, x, label):
        out = self.forward(x) # 正向传播
        self.loss = self.mse(out, label) # 根据正向传播计算损失
        self.optim.zero_grad() #梯度清零
        self.loss.backward() # 计算梯度
        self.optim.step() # 应用梯度更新参数
    
    def getLoss(self):
        return self.loss
    
    def test(self, test_):
        return self.fc(test_)

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
    test_losses=typing.List[float],
    train_accuracies=typing.List[float],
    test_accuracies=typing.List[float],
    all_acc=float,
    all_loss=float,
    model_state=PythonPickledFile,
)

def predictAcc(model, data, label):
    # import torch
    out=model.test(torch.from_numpy(data).float())
    prediction = torch.max(out, 1)[1] # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    test_y = label.reshape(1,-1)
    target_y =torch.from_numpy(test_y).long().data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    return accuracy, model.getLoss().item()
    
@task(
    retries=2,
    cache=False,
    cache_version="1.0",
    requests=Resources(gpu=gpu, mem=mem, storage=storage),
    limits=Resources(gpu=gpu, mem=mem, storage=storage),
)
def pytorchIrisTask(hp: Hyperparameters) -> TrainingOutputs:
    
    
    data, labels, train, test, train_label, test_label = loadData()
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(), torch.from_numpy(train_label).long())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=hp.batch_size, shuffle=True)
    
    model = Model()
    
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    # train
    for epoch in range(hp.epochs):
        for step, (x, y) in enumerate(train_loader):
            y = torch.reshape(y, [hp.batch_size])
            model.train(x, y)
            # wandb.watch(model)
            if epoch % 20 == 0:
                accuracy, loss = predictAcc(model, test, test_label)
                # print("莺尾花测试集预测准确率", accuracy)
                print(f"Epoch: {epoch} | Step: {step} | Loss: {model.getLoss().item()} | acc: {accuracy}")
                # wandb.log({"loss": loss, "epoch":epoch, "accuracy": accuracy})
        
        train_acc, train_loss = predictAcc(model, train, train_label)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # test loss
        test_acc, test_loss = predictAcc(model, test, test_label)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        

    # test for all data
    all_acc, all_loss = predictAcc(model, data, labels)
    
    
    # save model
    model_file = "iris_model.pt"
    torch.save(model.state_dict(), model_file)
    return TrainingOutputs(
        train_losses=train_losses,
        test_losses=test_losses,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        all_acc=all_acc,
        all_loss=all_loss,
        model_state=PythonPickledFile(model_file),
    )

@workflow
def pytorchTrainingWorkflow(
    hp: Hyperparameters = Hyperparameters(epochs=100, batch_size=10)
) -> TrainingOutputs:
    a = 1
    return pytorchIrisTask(hp=hp)

if __name__=='__main__' :
    train_losses, test_losses, train_accuracies, test_accuracies, all_acc, all_loss, model = pytorchTrainingWorkflow(hp=Hyperparameters(epochs=100, batch_size=10, learning_rate=0.01))
    # print(f"Model: {model}, Accuracies: {accuracies}, All acc: {acc}")
    print(train_losses, test_losses, train_accuracies, test_accuracies, all_acc, all_loss, model)