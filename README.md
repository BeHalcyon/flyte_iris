# iris_project

A template for the recommended layout of a Flyte enabled repository for code written in python using [flytekit](https://docs.flyte.org/projects/flytekit/en/latest/).

## Usage

To get up and running with your Flyte project, we recommend following the
[Flyte getting started guide](https://docs.flyte.org/en/latest/getting_started.html).


## NOTE
1. This APP name is also added to ``docker_build_and_tag.sh`` - ``APP_NAME``
2. We recommend using a git repository and this the ``docker_build_and_tag.sh``
   to build your docker images
3. We also recommend using pip-compile to build your requirements.

## EXPERIMENT
本笔记记录采用flyte搭建简易机器学习平台。默认已安装docker、flyte、python3.7。实验操作系统：Ubuntu 18.04 amd64

分为以下几个步骤：
##### 搭建flyte平台
关于flyte平台介绍，参考[该博客](https://xiangyanghe.cn/articles/292 "该博客")
###### 初始化阶段
0. 初始化一个flyte项目目录：
```bash
flytectl init iris_project
```

1. 进入该目录，并开启sandbox，其中，--source后将指定的目录作为拟生成flyte平台的根目录，该步骤耗费大约5分钟：
```bash
cd /path/to/flyte/project && flytectl sandbox start --source iris_project
```

###### 开发阶段
Flyte使用Docker容器来打包workflow和task，并将它们发送到远程的Flyte集群。上述的项目目录中已经包含了一个Dockerfile。在开放中，首先构建Docker容器，并将构建的image推送到注册表中。

2. flyte-sandbox在Docker容器中本地运行，所以不需要推送Docker镜像。通过简单地在flyte-sandbox容器中构建映像来组合构建和推送步骤。 这可以通过以下命令实现：
```bash
flytectl sandbox exec -- docker build . --tag "iris:v1"
```

3. 接下来，使用与flytekit绑定的pyflyte cli打包workflow，并将其上传到Flyte后端。该映像与前一步中构建的映像相同：
```bash
pyflyte --pkgs flyte.workflows package --image "iris:v1" --force
```

4. 将上述的包上传（注册）到Flyte后端。这里的版本v1不需要与上面命令中使用的版本匹配，但是通常建议与版本匹配，这样更容易跟踪：
```bash
flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v1
```

##### 实现iris分类模型
上述步骤初始化并部署了flyte平台，本节将iris模型部署上去。
- pytorch版

```python
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
```
```
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
```
```
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
```

- tensorflow版

```python
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
```

##### 将iris分类模型部署到flyte平台上

代码中导入了sklearn, pytorch，tensorflow等包，需要同样导入到docker中，所以需要修改Dockerfile，如下：

```bash
FROM python:3.7-slim-buster

WORKDIR /root
# 原始为/opt/venv，安装错误，修改为root目录下
ENV VENV /root/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

# ENV VENV /opt/venv
# Virtual environment
# 先激活虚拟环境，再安装库
RUN python3 -m venv ${VENV}

# RUN chmod +x ${VENV}/bin/activate && ${VENV}/bin/activate

ENV PATH="${VENV}/bin:$PATH"

RUN apt-get update && apt-get install -y build-essential

# Install the AWS cli separately to prevent issues with boto being written over
RUN pip3 install awscli
# Similarly, if you're using GCP be sure to update this command to install gsutil
# RUN curl -sSL https://sdk.cloud.google.com | bash
# ENV PATH="$PATH:/root/google-cloud-sdk/bin"

# RUN pip3 install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# 安装torch
RUN python -m pip --no-cache-dir install --upgrade --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# RUN pip3 install wandb
# 安装sklearn
RUN pip3 install sklearn

# Install Python dependencies
COPY ./requirements.txt /root
RUN pip install -r /root/requirements.txt

# Copy the actual code
COPY . /root

# This tag is supplied by the build script and will be used to determine the version
# when registering tasks, workflows, and launch plans
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
```
在此之后，重新执行“开发阶段”，注意tag的版本。

##### 实验结果（以pytorch结果为例）

浏览器中输入http://localhost:30081/console ，进入flytesnacks项目的development domain。
<div algin="center">
<img src="http://xiangyanghe.cn/wp-content/uploads/2022/03/a24ce5741a5257a6b3a51945a4e8f84.png" />
</div>

执行对应的workflow，结果如下：
<div algin="center">
<img src="http://xiangyanghe.cn/wp-content/uploads/2022/03/ef34448a7fd7067d739067ea5808212.png"/>
</div>

<div algin="center">
<img src="http://xiangyanghe.cn/wp-content/uploads/2022/03/ef34448a7fd7067d739067ea5808212.png" />
</div>


##### 将容器打包为image并上传（当前有效为v4.0版，未经测试）

- 上dockerhub注册用户。个人注册用户名为xiangyanghe，邮箱为xiangyanghe@zju.edu.cn
- 执行命令，查询container_id：
```bash
docker ps
```
- 登录dockerhob
```bash
docker login
```
- commit当前容器id，打上tag
```bash
docker commit <container_id> xiangyanghe/iris_flyte_project:v4.0
```
- commit好像没有打包成功，用build实现
```bash
docker build . -t xiangyanghe/iris_flyte_project:v4.0
```
- 上传到dockerhub中
```bash
docker push xiangyanghe/iris_flyte_project:v4.0
```
上传地址为: https://hub.docker.com/repository/docker/xiangyanghe/iris_flyte_project
 
##### 服务器不在本地时的操作
实现将刚刚上传的镜像进行远程访问，其flyte服务端不在本地，而在远程，因此，需要配置远程服务器：

```bash
flytectl config init --host='10.76.3.83:30081' --insecure
```
在开发好程序后，将目录内的文件封装为docker image：
```bash
docker build . --tag <registry/repo:version>
docker push <registry/repo:version>
```
随后将指定目录导出：
```bash
pyflyte --pkgs flyte.workflows package --image <registry/repo:version>
```
最后将生成的package注册到服务器中。

以下操作为拉取上述的image并直接运行docker，未经验证。
```bash
// 直接拉image，需要进一步创建容器
docker pull xiangyanghe/iris_flyte_project:v1.0
```
或者直接通过打包好的image实现（有待验证）
```bash
docker run --rm --privileged -p 30081:30081 -p 30084:30084 -p 30088:30088 docker.io/xiangyanghe/iris_flyte_project:v4.0
```
由此，可实现镜像迁移，跨设备一键部署。

##### 其他注意事项

打包的docker images 过多引起存储设备不足，可以通过以下命令删除所有的容器：
```bash
docker system prune -a --volumes
```

