## 一、PASS环境配置

源代码地址：[下载地址](https://github.com/Impression2805/CVPR21_PASS)

对环境无要求，使用任意CUDA以及对应版本的torch应该都能运行，提示找不到库安装即可

代码使用的数据集为CIFAR-100，由于官网下载较慢，最好提前下载压缩包，在源代码下创建dataset文件夹存放

## 二、PASS运行

### 2.1 代码参数

```python
# main.py
parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
# 指定模型训练时的batch size
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
# 指定训练用的数据集
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
# 指定数据集中类的总数量
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
# PASS的训练，需要先使用数据集中一半的类训练一个base model，再增量地学习
# 因此需要指定base model使用的类数量
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
# 还需要指定增量学习的任务数量，对应论文中的phase
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
# 损失函数中原型增强部分的权重
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
# 损失函数中知识蒸馏部分的权重
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
```

### 2.2 准备工作

```python
# main.py
# 要使用的GPU信息
cuda_index = 'cuda:' + args.gpu
device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
# 用数据集剩余的类数量(cifar-100是50个)和训练中的任务数量(默认是10)计算每个任务需要的类数量task_size
task_size = int((args.total_nc - args.fg_nc) / args.task_num)
# 用2.1中的参数生成模型存储路径
file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)
```

### 2.3 特征提取器创建

```python
# main.py
feature_extractor = resnet18_cbam()
```

该语句将会调用如下函数：

```python
# ResNet.py
def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model
```

就是PyTorch提供的ResNet网络创建模块，论文指出使用ResNet18框架从头开始训练，因此不需要加载预训练模型

<font color='red'>作者更改过其中forward的内容，该ResNet将直接输出特征提取后flatten的结果，而不会经过线性层</font>

### 2.4 PASS模型创建

```python
# main.py
model = protoAugSSL(args, file_name, feature_extractor, task_size, device)
```

该语句创建了一个**protoAugSSL**实例，详见第三章

### 2.5 模型训练

```python
# main.py
for i in range(args.task_num+1):
    if i == 0:
        old_class = 0
    else:
        old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
    model.beforeTrain(i)
    model.train(i, old_class=old_class)
    model.afterTrain()
```

进行增量学习训练，训练的轮次是增量学习任务数量+1，原因是第一个任务是使用cifar-100的50个类训练一个base model，后续则为增量学习任务

一次训练分为三个部分，**beforeTrain**、**train**以及**afterTrain**

## 三、protoAugSSL

```python
# PASS.py
class protoAugSSL:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        '''
        输入参数：
        	-args				全局参数变量
        	-file_name			存储路径
        	-feature_extractor	创建的特征提取器
        	-task_size			每个任务中类的数量
        	-device				使用的CPU/GPU
        主要流程：
        	1、类内变量设置
        	2、self.model = network(args.fg_nc*4, feature_extractor)，创建模型，来自myNetwork，详见第四章
        		2.1 创建的类数量是args.fg_nc*4的原因，是因为作者使用了自监督策略，一个类能够生成额外三个新类，因此分类器应该扩大成原来的4倍
        	3、设置对训练集和测试集的变换self.train_transform和self.test_transform
        	4、通过iCIFAR100类创建训练数据集以及测试数据集，详见第五章
        '''
    def beforeTrain(self, current_task):
        '''
        该部分主要用于获取训练中使用的训练集以及测试集
        主要流程：
        	1、确认任务列表(存在current_task是否为第一个任务的判断)
        	2、根据任务列表调用self._get_train_and_test_dataloader()来得到对应训练集和测试集的loader、
        	3、如果当前任务编号不是第一个任务，则需要调用self.model.Incremental_learning()进行增量学习设置
        '''
    def _get_train_and_test_dataloader(self, classes):
        '''
        主要流程：
        	1、调用iCIFAR100类中的getTrainData()和getTestData()方法获得指定的训练集和测试集
        	2、使用torch提供的DataLoader类封装为数据加载器
        	3、返回两个数据加载器
        '''
    def _get_test_dataloader(self, classes):
    def train(self, current_task, old_class=0):
        '''
        输入参数：
        	-current_task	当前任务编号
        	-old_class		旧任务数量
        主要流程：
        	1、设置模型优化器，使用Adam，学习率由之前的参数设置中获取，同时使用了学习率衰减策略
        	2、进入训练流程 for epoch in range(self.epochs):
        		2.1 迭代器获取数据及标签
        		2.2 自监督学习设置，利用torch.rot90()对图像进行3次旋转，形成3个新的类，并生成新的标签
        		2.3 使用self._compute_loss()将训练数据输入到网络中获取损失
        		2.4 反向传播流程
        	3、调用self.protoSave()存储原型
        '''
    def _test(self, testloader):
    def _compute_loss(self, imgs, target, old_class=0):
        '''
        输入参数：
        	-imgs		图像数据
        	-target		标签信息
        	-old_class	旧任务数量
        主要流程：
        	1、图像输入到网络中获取输出
        	2、计算一个分类损失(交叉熵损失)
        	3、如果不存在旧模型(其实就是第一个任务)，就直接返回分类损失
        	4、如果存在旧模型，就有以下步骤：
        		4.1 使用当前的图像，分别从self.model和self.old_model中提取图像特征(仅经过特征提取器)
        		4.2 使用torch.dist()计算新旧特征之间的距离，当作知识蒸馏的损失使用，确保特征提取器不会有太大的变化
        		4.3 随机提取旧类的原型，并进行增强
        		4.4 将增强后的类原型输入到模型的分类器层中，得到结果
        		4.5 将结果与原型对应的标签比对，作为原型增强损失
        		4.6 将当前任务分类损失+原型增强损失+知识蒸馏损失作为组合结果返回
        '''
    def afterTrain(self):
        # 保存模型到指定路径，并且将其加载为旧模型self.old_model用于下一次任务
    def protoSave(self, model, loader, current_task):
        '''
        输入参数：
        	-model			本次任务训练模型
        	-loader			数据加载器
        	-current_task	当前任务编号
        主要流程：
        	1、所有图像输入到模型的特征提取器中获取特征图
        	2、各特征以标签为依据进行类的整合
        	3、计算每个类的均值，如果是第一个任务的话，还需要计算一个原型增强时的参数
        	4、存储计算出的原型以及它的类标签
        '''
```

## 四、network

```python
# myNetwork.py
class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        '''
        输入参数：
        	-numclass			类数量(分类器头数量)
        	-feature_extractor	创建的特征提取器
        主要流程：
        	在特征提取器之后增加一个线性层
        '''
    def forward(self, input):
    def Incremental_learning(self, numclass):
        '''
        增量学习策略，在新任务到来后扩充原有的分类器
        主要流程：
        	1、保存原分类器的相关权重以及输入输出神经元数量
        	2、根据输入的numclass创建新的线性层
        	3、将原分类器的部分权重导入到新的分类器中
        '''
    def feature_extractor(self, inputs):
        # 数据仅通过特征提取器获取特征
```

## 五、iCIFAR100

该类继承于torchvision提供的**CIFAR100**类，增加了为满足增量学习的相关设置，对一些函数进行了重写

```python
# iCIFAR100.py
class iCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 test_transform=None, target_test_transform=None, download=False):
        '''
        输入参数：
        	-root					数据集文件地址
        	-train					指示是训练集还是测试集
        	-transform				原CIFAR100中需要指示的参数，对训练集的图形变换
        	-target_transform		原CIFAR100中需要指示的参数，默认为None
        	-test_transform			原CIFAR100中需要指示的参数，对测试集的图形变换
        	-target_test_transform	原CIFAR100中需要指示的参数，默认为None
        	-download				指示是否需要在线下载数据集
        主要流程：
        	本质还是CIFAR100类，只是增加了一些变量
        增加：
        	self.target_test_transform = target_test_transform
        	self.test_transform = test_transform
        	self.TrainData = []
        	self.TrainLabels = []
        	self.TestData = []
        	self.TestLabels = []
        '''
    def concatenate(self, datas, labels):
        # 字面意思，对列表进行降维重组
    def getTestData(self, classes):
        # 与getTrainData同理，区别在于测试集需要记录从整个训练开始到当前任务的全部类的测试数据
        # 存于self.TestData以及self.TestLabels中
    def getTestData_up2now(self, classes):
    def getTrainData(self, classes):
        '''
        输入为任务中包含的类的列表，最终这些类的所有图像数据和标签信息存储于self.TrainData和self.TrainLabels
        这样self.TrainData就不再为空，__getitem__就可以判定当前类为训练集实例，而调度相应的方法
        主要流程：
        	1、声明两个临时列表分别用于存储TrainData以及TrainLabel
        	2、根据classes类别获取数据
        		2.1 self.data是CIFAR100这个类存储CIFAR-100数据集数据的变量，而继承于该类的iCIFAR100也有这个变量
        		2.2 同时self.target存储了一一对应的标签，利用这个标签信息就可以筛选出self.data中需要的图像数据
        	3、调用self.concatenate()对列表重组一下(降维)，然后赋值给self.TrainData和self.TrainLabels
        	4、控制台显示信息
        '''
    def getTrainItem(self, index):
        '''
        用于给__getitem__调度的训练数据获取函数
        主要流程：
        	1、从self.TrainData以及self.TrainLabels中获得对应index的图像数据以及标签，图像需要转为narray
        	2、根据是否存在图像以及标签的transform设置来进行相应的变换
        	3、返回index，数据以及标签
        '''
    def getTestItem(self, index):
        # 用于给__getitem__调度的测试数获取函数，流程和getTrainItem()相同
    def __getitem__(self, index):
        '''
        更改了原有方法，通过self.TrainData和self.TestData的存在进行判断
        分别调度self.getTrainItem()以及self.getTestItem()
        也验证了一些想法，作为Dataloader的参数，Dataset只要提供__getitem__以及__len__就可以运作
        前者为获取数据提供支持，后者则为Dataloader确认迭代器循环次数提供支持
        '''
    def __len__(self):
        # 返回self.TrainData或是self.TestData的长度
    def get_image_class(self, label):
```
