## 一、La-MAML配置

### 1.1 源代码下载

[github](https://github.com/montrealrobotics/La-MAML)

### 1.2 数据集准备

- <b><font color='red'>Tiny-Imagenet-200</font></b>

配置La-MAML需要准备的数据集之一，Tiny ImageNet Dataset源于ImageNet，其包含有200个类别，每个类别含有500张训练突袭，50张验证图像和50张测试图像，其分辨率均为64${\times}$64

[下载地址](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

也可以运行源代码中的脚本```download_tinyimgnet.sh```获取

```bash
# download_tinyimgnet.sh
echo "Downloading Data..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping Data..."
unzip tiny-imagenet-200.zip
echo "Last few steps..."
rm -r ./tiny-imagenet-200/test/*
python3 val_data_format.py
find . -name "*.txt" -delete
```

如果不是用脚本完成的获取，则需要继续按照脚本的指令顺序完成步骤，就是解压数据集-删除test下所有文件-运行```val_data_format.py```，当然该文件中的文件地址都需要替换成本地Tiny-Imagenet-200的文件地址

- <b><font color='red'>CIFAR-100</font></b>

直接使用了torchvision提供的数据集CIFAR100，如果不想在线下载数据集的话，可以先在官网下载python版本的数据集，然后放置在指定的文件夹下

### 1.3 运行环境

直接按照源代码中```README.md```的提示就行

```bash
conda env create -f environment.yml
conda activate lamaml
pip install -r requirements.txt
```

安装完包以后就可以启动环境了

## 二、LA-MAML运行

### 2.1 基本流程

运行La-MAML以及源代码提供的其他的连续学习方法，最好使用代码中提供的命令行(```run_experiments.sh```)运行

La-MAML的基本流程如下：

- 声明数据集加载器

  ```loader = Loader.IncrementalLoader(args, seed=args.seed)```

- 声明网络模型

  ```model = Model.Net(n_inputs, n_outputs, n_tasks, args)```

- 开始进行训练

  ```life_experience(model, loader, args)```

### 2.2 数据集加载器

- <b><font color='red'>Class Incremental Loader & Tiny ImageNet</font></b>

```python
# ./dataloaders/class_incremental_loader.py
class IncrementalLoader:
    def __init__(self, opt, shuffle=True, seed=1):
        '''
        输入参数：
        	-opt		全局环境变量
        	-shuffle	指示是否需要打乱数据
        	-seed		用于给random使用的seed
        主要流程：
        	1、必要变量的获取
        	2、调用同文件下idataset.py提供的_get_datasets()获取数据集类(该情况下是dataloaders.idataset.iImgnet)
        	3、作为输入调用self._setup_data()生成数据集
        	4、一些类内变量的设置
        	5、调用self._setup_test_tasks()生成测试任务，会有self.test_tasks以及self.val_tasks
        '''
    def n_tasks(self):
        # 返回任务数量
    def new_task(self, memory=None):
        # 决定本次任务所涉及的类别，同时调用self._get_loader()将其封装为Dataloader返回
    def _setup_test_tasks(self, validation_split):
    def get_tasks(self, dataset_type='test'):
        # 用于获取测试任务以及预测任务
    def get_dataset_info(self):
        # 返回数据集以及训练设置相关的信息
    def _select(self, x, y, low_range=0, high_range=0):
        # 获取选定的数据
    def _get_loader(self, x, y, shuffle=True, mode="train"):
        # 结合对应的图像变换，将选定的数据封装到Dataloader中，并将加载器返回
    def _setup_data(self, datasets, class_order_type=False, seed=1, increment=10, validation_split=0.):
        '''
        输入参数：
        	-datasets			设置使用的数据集，这里就是代表TinyImageNet的iImgnet类
        	-class_order_type	设置数据集的类的打乱方式，["random", "chrono", "old", "super"]，一般用'random'就行
        	-seed				设置random中的seed
        	-increment			设置连续学习任务任务中类的数量
        	-validation_split	训练集分离，该参数设置了训练集分离成训练集和验证集的比例，论文预设的是0.1
        主要流程：
        	代码设计似乎是可以使用复数数据集的，这里仅有TinyImageNet因此暂不考虑这层逻辑
        	1、从环境变量中获得数据集地址，这个需要在脚本中更改
        	2、使用给定的dataset类中的base_dataset获取训练集与数据集，详情请看第3章
        		2.1 数据集的构成为[('/data/data-user-njf8...537_0.JPEG', 0),...]，位于变量samples中
        		2.2 数据以元组为单元，此时仅获得了图像的地址，并指示标签的index
        		2.3 训练集与测试集分别存于train_dataset和test_dataset中
        	3、获取训练数据以及对应标签，(x_train, y_train)
        	4、调用self._list_split_per_class()将训练集分离成训练集和验证集
        	5、获取测试集数据集以及标签
        	6、随机打乱类别
        		6.1 因为Tiny-ImageNet中前后文件夹对应的类别存在一定联系，比如海洋生物之间会倾向于放在一起
        		6.2 因此需要再次打乱顺序，使这种联系性失效
        		6.3 打乱后标签需要重新映射给对应数据
        	7、最终获得以下数据：
        		-self.data_train
        		-self.targets_train
        		-self.data_val
        		-self.targets_val
        		-self.data_test
        		-self.targets_test
        '''
    def _map_new_class_index(y, order):
        # 用于类别标签的重新映射
    def _split_per_class(x, y, validation_split=0.):
        '''
        输入参数：
        	-x					图像数据
        	-y					对应的标签数据
        	-validation_split	指示分离比例
        将训练数据按照指定的比例分离成一个训练数据子集和一个验证数据子集，对每个类进行单独分离
        按照0.1的比例设置的话，Tiny-ImageNet数据集训练集中每个类包含500张图像，分离为450张用于训练50张用于验证
        返回x_val, y_val, x_train, y_train
        '''
    def _list_split_per_class(x, y, validation_split=0.):
    def get_idx_data(self, idx, batch_size, mode="test", data_source="train"):
    def get_custom_loader(self, class_indexes, mode="test", data_source="train"):
```

其他的数据集事实上也可以用这样的方法完成，只要提供能够对接上的方法就可以了

### 2.3 网络模型

实验中训练的网络模型称为“**pc_cnn**”，它是一个四层卷积层作为特征提取器，后接一个全连接层作为分类器的分类网络

其的声明发生在lamaml_base的```__init__```中，详细可以查看**2.4**的部分

读过论文可以知道，lamaml存在一个自学习的学习率，而其运作的方式就是将这个学习率也加入到反向传播的过程中

网络由两个部分产生，分别是```learner.py```以及```modelfactory.py```，类似的结构其实在maml的源代码中也有使用

```modelfactor.py```设置网络的相关配置信息

```python
# ./model/meta/modelfactory.py
class ModelFactory():
    @staticmethod
    def get_model(model_type, sizes, dataset='mnist', args=None):
        if model_type == 'pc_cnn':
            channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [640, 640], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]
```

```learner.py```则是调度网络的生成以及前馈传播的进行

```python
# ./model/meta/learner.py
class Learner(nn.Module):
    def __init__(self, config, args = None):
        # 按照输入的config对网络中各层参数进行创建和初始化
    def extra_repr(self):
    def forward(self, x, vars=None, bn_training=False, feature=False):
        '''
        输入参数：
        	-x				输入的训练/测试数据
        	-vars			外部输入的权重，如果有的话就使用这个输入的权重计算前馈传播，否则就使用网络自己的参数计算
        	-bn_training	未使用过该参数，效果未知
        	-feature		未使用过该参数，效果未知
        该部分设计了各种神经网络组件的运算逻辑，以完成前馈传播流程
        '''
    def zero_grad(self, vars=None):
        # 与nn.Moudle的zero_grad相同，也是执行梯度清零
    def define_task_lr_params(self, alpha_init=1e-3): 
        # 相当于复制了当前的网络创造了一个小网络，并且参数全部初始化为alpha_init
    def parameters(self):
        # 返回网络内权重参数，因为继承自nn.Moudle所以作用和nn.Module是一致的
```

这样的设计结构是使自学习学习率能够正确运作的保证

### 2.4 训练

训练的流程位于```main.py```之中

```python
# main.py
def life_experience(model, inc_loader, args):
    # 根据数据加载器提供的任务数量决定训练的迭代次数
    for task_i in range(inc_loader.n_tasks):
        # 调用加载器的new_task()函数获取本次的任务数据加载器train_loader
        task_info, train_loader, _, _ = inc_loader.new_task()
        for ep in range(args.n_epochs):
            prog_bar = tqdm(train_loader)
            # 迭代器运作
            for (i, (x, y)) in enumerate(prog_bar):
                v_x = x
                v_y = y
                if args.cuda:
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()
                model.train()
                # 调用模型的observe()函数开始训练
                loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])
```

- <b><font color='red'>lamaml_cifar</font></b>

```python
# ./model/lamaml_cifar.py
class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        # 初始化主要集中于它的父类BaseNet
    def take_loss(self, t, logits, y):
        # inner_update也就是内循环更新所使用的损失计算策略
    def take_multitask_loss(self, bt, t, logits, y):
        # meta_loss也就是外循环更新所使用的损失计算策略
    def forward(self, x, t):
        # 正常的前向传播，不过也依据当前任务标识，对不相关的分类头输出进行截断
    def meta_loss(self, x, fast_weights, y, bt, t):
        # 按照不同的任务标识计算对应不同的loss
    def inner_update(self, x, fast_weights, y, t):
        '''
        输入参数：
        	-x				输入数据
        	-fast_weight	临时权重，用于内循环快速更新
        	-y				对应数据标签
        	-t				任务标识
        主要流程：
        	1、计算任务标识对应的分类头偏置(利用self.compute_offsets())
        	2、由于网络的特殊性，可以将数据x以及临时权重fast_weight输入网络进行前前馈传播获得预测结果
        		2.1 如果fast_weight是None的话，就用网络本身的参数进行计算
        	3、如果此时的fast_weight为None，就将网络自身参数赋值给它
        	4、使用torch.autograd.grad计算梯度
        	5、利用论文算法计算更新后的权重fast_weight
        	6、返回fast_weight用于计算meta_loss以及下一次的inner_update
        '''
    def observe(self, x, y, t):
        '''
        输入参数：
        	-x	训练数据
        	-y	数据对应标签
        	-t	当前任务标识
        进行lamaml所设计的训练流程的所在
        主要流程如下：
        	1、for pass_itr in range(self.glances):
        		1.1 这是训练所使用的其中一个策略，glances指定了循环的次数，同一组数据将有机会与memory中不同数据进行组合，减缓灾难性遗忘
        		1.2 每次开始时都需要打乱输入的数据的顺序以及标签，一般glances可以设置成2或者1，也存在设置成10
        		1.3 self.M = self.M_new.copy()，self.M是使用在训练中的存储器，而新的数据则会保存在self.M_new中
        		1.5 用当前数据和self.M中随机抽取一定数量的数据进行组合，得到bx, by, bt
        		1.4 数据进一步分离，这里分成了5个部分
        		1.5 一部分一部分输入到网络中进行inner_update()
        		1.6 调用push_to_mem()将数据存储入self.M_new中
        		1.7 进行过一次inner_update()后还需要进行一次meta_loss()的计算，以列表分存
        	2、对meta_losses进行处理
        	3、对学习率网络进行更新
        	4、再用学习率网络组织对本体网络的更新(异步更新)
        	5、返回一些信息
        '''
```

- <b><font color='red'>lamaml_base</font></b>

```python
# ./model/lamaml_base.py
class BaseNet(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        '''
        输入参数：
        	-n_inputs	输入尺寸
        	-n_outputs	输出尺寸
        	-n_tasks	任务数量
        	-args		环境变量
        主要流程：
        	1、调用ModelFactory的get_model()方法得到网络配置，详情可见2.3部分
        	2、使用Learner类创建网络模型，存为self.net，详情可见2.3部分
        	3、调用self.net也就是Learner类的define_task_lr_params()方法创建一个学习率网络
        	4、分别为主体网络模型和学习率网络模型创建优化器
        	5、设置损失函数，一般为分类任务的交叉熵损失
        	6、进行一些连续学习以及深度学习的相关参数设置
        '''
    def push_to_mem(self, batch_x, batch_y, t):
        # 将任务内数据存储到self.M_new中，采用了蓄水池抽样法的策略，保证了在记忆区域中每个类都能拥有数据
    def getBatch(self, x, y, t, batch_size=None):
        # 从记忆区域中随机抽取数据，与输入的x，y组成组合数据的方法
    def compute_offsets(self, task):
        '''
        用于计算偏置，输入为任务的编号
        以Tiny-ImageNet为例，共计200个类，如果以5个类作为一个任务，那么共有40个任务，同时模型是一个200类的分类模型
        而当进行当前任务的时候，并不需要考虑其他的类的分类结果，因此可以只计算当前类所属分类头的loss而不用计算其他的分类头
        于是就有了这个计算偏置的函数，用于计算当前所属的分类头
        '''
    def zero_grads(self):
        # 对类内所有网络以及优化器进行梯度清零
```

## 三、数据集及处理

### 3.1 DummyDataset

```python
# ./dataloaders/idatasets.py
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, trsf, pretrsf = None, imgnet_like = False, super_y = None):
        '''
        输入参数：
        	-x				图像数据
        	-y				对应数据标签
        	-trsf			图像变换，用于图像转变为imagearray之后
        	-pretrsf		前置图像变换，用于图像转变为imagearray之前
        	-imgnet_like	指示数据是否来自于ImageNet/Tiny-ImageNet(这俩本质一样)
        	-super_y		默认为None
        继承了torch.utils.data.Dataset，需要对__len__()以及__getitem__()进行实现
        '''
    def __len__(self):
        # 返回数据集的规模
    def __getitem__(self, idx):
        '''
        主要流程：
        	1、根据idx抽取self.x以及self.y中的数据和标签
        	2、用self.pretrsf()对图像进行图像变换
        	3、如果self.imgnet_like为False的话，需要使用Image.fromarray()将图像变为array
        	4、用self.trsf()再进行图像变换
        	5、返回图像以及标签
        '''
```

### 3.2 iImgnet

```python
# ./dataloaders/idatasets.py
class iImgnet(DataHandler):
    # 设置了base_dataser，实质上是torchvision提供的ImageFolder，其通过输入数据集所在地址以其下文件夹为一个类获取图像数据
    # 因此在准备数据集时也需要设计，而文件夹名称就直接作为标签来使用了，同时会给一个按照文件夹读取顺序的index来对照这个标签
    base_dataset = datasets.ImageFolder
    # 一些对图像的变换设置
    top_transforms = [
        # TinyImageNet的前置图像变换，实际就是读取图像为RGB格式
        lambda x: Image.open(x[0]).convert('RGB'),
    ]
    train_transforms = [
        # 额外用于训练集的图像变换
        transforms.RandomCrop(64, padding=4),           
        transforms.RandomHorizontalFlip()
    ]
    common_transforms = [
        # 通用的图像变换
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
    ]
    # 类别，一般用不到
    class_order = [                                                                     
        i for i in range(200)
    ]
```



