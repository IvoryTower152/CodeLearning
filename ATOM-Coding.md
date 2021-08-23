## 一、运行环境

参考**PyTracking**代码中的```INSTALL.md```或者```INSTALL_win.md```进行环境配置，后者用于在windows环境下配置运行环境，不建议在windows下配置

一般情况下能顺利配置，Linux下需要一些root权限，同时源代码还需要加入**pr pooling**的模块，加入方法在源代码给出的文件中有说明

**源代码地址**：[地址](https://github.com/visionml/pytracking)

**Pr-Pooling地址**：[地址](https://github.com/SirLPS/roi_pooling)

## 二、数据集

ATOM的训练需要**LaSOT**、**GOT-10K**、**TrackingNet**、**COCO**数据集，需要提前准备

## 三、训练流程—数据集

由```./ltr/run_training.py```进入

直接跳到```./ltr/train_settings/bbreg/atom.py```开始

首先是对训练流程进行配置

```python
settings.description = 'ATOM IoUNet with default settings, but additionally using GOT10k for training.'
settings.batch_size = 64
settings.num_workers = 8
settings.print_interval = 1
settings.normalize_mean = [0.485, 0.456, 0.406]
settings.normalize_std = [0.229, 0.224, 0.225]
settings.search_area_factor = 5.0
settings.feature_sz = 18
settings.output_sz = settings.feature_sz * 16
settings.center_jitter_factor = {'train': 0, 'test': 4.5}
settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
```

然后是训练及验证数据的初始化


```python
# train
lasot_train = Lasot(settings.env.lasot_dir, split='train')
got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
coco_train = MSCOCOSeq(settings.env.coco_dir)
# val
got10k_val = Got10k(settings.env.got10k_dir, split='votval')
```

主要目的是为了得到各个数据集的信息，均基于类**BaseVideoDataset**，而BaseVideoDataset继承于**Dataset**

该部分其实是为**5.1**和**5.2**的数据采样器准备的

### 3.1 LaSot

```python
# ./ltr/dataset/lasot.py
class Lasot(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
    def _build_sequence_list(self, vid_ids=None, split=None):
    def _build_class_list(self):
    def get_name(self):
        # 返回数据集名称
    def has_class_info(self):
        # return true
    def has_occlusion_info(self):
        # return true
    def get_num_sequences(self):
        # 认为效果等同于__len__，返回训练集视频数量
    def get_num_classes(self):
        # 返回数据集class的数量
    def get_sequences_in_class(self, class_name):
        # 返回属于该类的视频序列列表
    def _read_bb_anno(self, seq_path):
        # 获取gt box
    def _read_target_visible(self, seq_path):
        # 获取含有目标的视频帧
    def _get_sequence_path(self, seq_id):
        # 获取序列的地址
    def get_sequence_info(self, seq_id):
        # 会返回一个字典{'bbox': bbox, 'valid': valid, 'visible': visible}，元素好像是张量，大概意思应该是返回有效的视频帧
    def _get_frame_path(self, seq_path, frame_id):
    def _get_frame(self, seq_path, frame_id):
    def _get_class(self, seq_path):
    def get_class_name(self, seq_id):
    def get_frames(self, seq_id, frame_ids, anno=None):
        '''
        按照输入的视频id以及要获取的帧id列表，由该函数返回对应的frames
        输出如下：
        	frame_list: 帧所组成的列表
        	anno_frames: 包含了bounding box的一些注释信息
        	object_meta: 这个视频序列的一些信息，一个字典
        '''
```

<font color='red'><b>初始化信息</b></font>

```python
# LaSot数据集包含了70个类，1120个视频(每个类16个视频)
# self.class_list中包含了所有类的类名
# self.class_to_id则是以字典的形式将所有的类用一个唯一的id(数字)表示
# self.sequence_list中包含了所有的视频序列名称
# self.seq_per_class将所有的类所属的序列进行了归纳 (此处可以利用，需先了解序列获取机理)
```

### 3.2 GOT-10k

```python
# ./ltr/dataset/got10k.py
class Got10k(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
    def get_name(self):
    def has_class_info(self):
    def has_occlusion_info(self):
    def _load_meta_info(self):
    def _read_meta(self, seq_path):
    def _build_seq_per_class(self):
    def get_sequences_in_class(self, class_name):
    def _get_sequence_list(self):
    def _read_bb_anno(self, seq_path):
    def _read_target_visible(self, seq_path):
    def _get_sequence_path(self, seq_id):
    def get_sequence_info(self, seq_id):
    def _get_frame_path(self, seq_path, frame_id):
    def _get_frame(self, seq_path, frame_id):
    def get_class_name(self, seq_id):
    def get_frames(self, seq_id, frame_ids, anno=None):
```

<font color='red'><b>初始化信息</b></font>

```python
# GOT-10k vottrain包含460个目标类别
# self.sequence_list将会包含全部序列名称 (9335个视频，经过split之后会变为7086个，但是每个类别的视频数量并不一致，最少为1，最多为1744)
# self.sequence_meta_info会以字典形式包含视频序列的一些属性 [object_class_name, motion_class, major_class, root_class, motion_adverb]
# self.seq_per_class将所有的类所属的序列进行了归纳
# self.class_list包含了所有类的类名
```

### 3.3 TrackingNet

```python
# ./ltr/dataset.tracking_net.py
class TrackingNet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, set_ids=None, data_fraction=None):
    def _load_class_info(self):
    def get_name(self):
    def has_class_info(self):
    def get_sequences_in_class(self, class_name):
    def _read_bb_anno(self, seq_id):
    def get_sequence_info(self, seq_id):
    def _get_frame(self, seq_id, frame_id):
    def _get_class(self, seq_id):
    def get_class_name(self, seq_id):
    def get_frames(self, seq_id, frame_ids, anno=None):
```

<font color='red'><b>初始化信息</b></font>

```python
# TrackingNet(0-3) 包含有21个目标种类，10044个视频序列，也存在分布不均衡，但是没有GOT-10K这么极端，视频基数较大
# self.sequence_list包含有所有的视频序列信息，以元组形式 (数据集编号，序列名称)
# self.seq_to_class_map是一个视频序列和对应的类别之间的一个映射表
# self.seq_per_class将所有的类所属的序列进行了归纳
# self.class_list包含了所有类的类名
```

### 3.4 COCO

```python
# ./ltr/dataset/coco_seq.py
class MSCOCOSeq(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2017"):
    def _get_sequence_list(self):
    def is_video_sequence(self):
    def get_num_classes(self):
    def get_name(self):
    def has_class_info(self):
    def get_class_list(self):
    def has_segmentation_info(self):
    def get_num_sequences(self):
    def _build_seq_per_class(self):
    def get_sequences_in_class(self, class_name):
    def get_sequence_info(self, seq_id):
    def _get_anno(self, seq_id):
    def _get_frames(self, seq_id):
    def get_meta_info(self, seq_id):
    def get_class_name(self, seq_id):
    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
```

<font color='red'><b>初始化信息</b></font>

```python
# COCO数据集包含80个类
# self.cats表明了每一个图像的对应信息
# self.class_list包含了所有类的名称
# self.sequence_list包含了所有图像id
# self.seq_per_class将所有的类所属的序列(单张图像)进行了归纳
```


## 四、训练流程—数据集处理
### 4.1 Transform模块

代码作者使用的Transfrom模块是由自己重写的

这个Transform功能相较于原本torchvision中的版本，不仅能对图像进行统一的Transform，还能对边界框(bbox)以及分割掩码(mask)进行同步的Transform，这在之前的torchvision.transforms中是无法做到的

通过设置输入的参数，可以使图像连同边界框以及掩码一起变换

```python
# Phase 1
transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))
```

上为```atom.py```中使用的一个Transform的代码

其中包含两个部分，首先是tfm.ToCrayscale，类形式如下

```python
# ./ltr/data/transforms.py
# 注释中说明该类的作用是按照概率将图像转变为灰度图
class ToGrayscale(TransformBase):
    def __init__(self, probability=0.5):
        # 进行一些参数的初始化
    def roll(self):
        # 如同字面意思，roll一个点，比概率小就返回True，大于就返回False
    def transform_image(self, image, do_grayscale):
        # 对图像进行变化
```

该类继承于一个父类TransformBase，其形式如下

```python
# ./ltr/data/transforms.py
# 注释中说明其为对象变换的基类，看起来像是作者参考原来的Transforms准备的
class TransformBase:
    def __init__(self):
        # 进行一些参数的初始化
    def __call__(self, **inputs):
        '''
        所有继承的子类都没有重新该函数，因此所有的Transform调用时会先调用这个函数
        主要行为是对输入的input进行元素分析，分别对"image"、"coords"、"bbox"、"mask"调用对应的transform_XXXX()函数
        '''
    def _get_image_size(self, inputs):
        # 很显然的字面意思
    def roll(self):
        # 需要继承的子类实现的抽象方法
    def transform_image(self, image, *rand_params):
        # 需要继承的子类实现的抽象方法
    def transform_coords(self, coords, image_shape, *rand_params):
        # 需要继承的子类实现的抽象方法
    def transform_bbox(self, bbox, image_shape, *rand_params):
        # 在TransformBase中已实现的方法，bbox的规格为[x, y, w, h]
    def transform_mask(self, mask, *rand_params):
        # 需要继承的子类实现的抽象方法
```

紧接```atom.py```中的那行代码，在tfm.ToCrayscale初始化完成后，会成为tfm.Transform的参数，类形式如下

```python
# ./ltr/data/transforms.py
'''
一组transform的组合，用于进行数据增强
按照输入，会顺序地执行继承于TransformBase的各种transform方法
一般会通过call函数启动流程
'''
class Transform:
    def __init__(self, *transforms):
        '''
        输入为一个transforms形参，这个参数应该和transforms.Compose()的性质差不多
        就是它可能包含一个transform行为也可能包含多个tranform行为
        后面的部分依旧是一些参数的初始化
        可知能用的输入应该有image, coords, bbox, mask
        能选择的操作应该有joint, new_roll
        '''
    def __call__(self, **inputs):
        '''
        调用Transform的时候就会调用这个函数
        inputs这个参数可以带的参数有：
        	image: 图像
        	coords: 2D图像坐标，不知道是啥，反正ATOM也没有用
        	bbox: 边界框信息[x, y, w, h]
        	mask: 类的分割掩码(Segmentation mask with discrete classes)
        还有两个指示参数可以携带：
        	joint[Bool]: 指明是否是联合变换，默认为True，如果为True，所有image/coords/bbox/mask都会进行完全相同的Transform
        	             否则每一个(image, coords, bbox, mask)为一个元组进行独立的Transform
        	new_roll[Bool]: 默认为True，若为True就会roll一个新的随机数，否则就不会，转而使用上一次保留的随机数
        '''
    def _split_inputs(self, inputs):
    def __repr__(self):
```

至此，```atom.py```中这一条语句就完成了，获得的就是一个变换器transform_joint，注释称这个变换器为<font color='red'><b>联合增广变换</b></font>

下一部分是<font color='red'><b>对训练数据的增强变换</b></font>

```python
# Phase 2
transform_train = tfm.Transform(
    tfm.ToTensorAndJitter(0.2),
    tfm.Normalize(mean=settings.normalize_mean,std=settings.normalize_std)
)
```

这里就能够印证之前的猜想，其向Transform输入了两个transform行为，则其接收的形参也会是一个复合的参数

应该和transforms.Compose()相同是按照顺序初始化，这两种新的变化代码如下

```python
# ./ltr/data/transforms.py
# 注释说明是用于张量变换以及图像亮度抖动
class ToTensorAndJitter(TransformBase):
    def __init__(self, brightness_jitter=0.0, normalize=True):
        # 对亮度抖动系数进行设置
    def roll(self):
    # 由于父类已经实现transform_bbox, 因此只要实现image和mask的部分
    def transform_image(self, image, brightness_factor):
    def transform_mask(self, mask, brightness_factor):
```

```python
# ./ltr/data/transforms.py
# 注释说明用于对图像进行归一化
class Normalize(TransformBase):
    def __init__(self, mean, std, inplace=False):
        # 对归一化均值和方差进行设置
    def transform_image(self, image):
```

就此针对训练数据的图像增强变换器就初始化完成了

下一部分是<font color='red'><b>对验证数据的增强变换</b></font>

```python
# Phase 3
transform_val = tfm.Transform(
    tfm.ToTensor(),
    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)
)
```

其也是两个变换按顺序组合

```python
# ./ltr/data/transforms.py
# 注释说明就是将其变换为张量, 等价于原先Transform中的toTensor()
class ToTensor(TransformBase):
    # 该类没有写init函数，直接使用父类的init函数
    def transform_image(self, image):
    def transfrom_mask(self, mask):
```

tfm.Normalize和之前相同，就不再重复说了

至此，所有的transform器声明完成，随着代码的调试，其功能也会陆续补上

### 4.2 对训练数据进行处理

```python
proposal_params = {
    'min_iou': 0.1,
    'boxes_per_frame': 16,
    'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]
}
data_processing_train = processing.ATOMProcessing(
    search_area_factor=settings.search_area_factor,
    output_sz=settings.output_sz,
    center_jitter_factor=settings.center_jitter_factor,
    scale_jitter_factor=settings.scale_jitter_factor,
    mode='sequence',
    proposal_params=proposal_params,
    transform=transform_train,
    joint_transform=transform_joint
)
```

对于proposal_params的参数含义目前不明

那么进入到processing.ATOMProcessing的部分

```python
# ./ltr/data/processing.py
'''
用于ATOM训练数据的处理, 图像将以以下方法处理：
1、通过添加一些噪声使目标边界框发生抖动
2、从图像中，裁剪出一个，以抖动后的边界框中心为中心，大小为search_area_factor的平方乘以抖动框的面积的区域
(按照论文以及一些视频来看，这个大小是5*5*边界框面积)
这个称作搜索区域
3、再抖动GTBox来获得一系列的proposals

抖动的目的是为了避免在学习过程中，目标总是位于搜索区域的中心
在搜索区域被切割出来后，将会按照output_sz进行尺寸的调整
'''
class ATOMProcessing(BaseProcessing):
    # init参数有点多这里就省略了
    def __init__():
        '''
        初始化中设置的有搜索区域的大小、输出的尺寸、抖动相关的参数，
        proposal相关的参数(proposal_params用在了这里)
        还有其他的参数
        '''
    def _get_jittered_box(self, box, mode):
        '''
        对输入的边界框进行抖动
        输入参数：
        	box: 输入的bounding box
        	mode: 选择'train'或是'test'用来指示数据类型
        '''
    def _generate_proposals(self, box):
        '''
        通过对输入的bounding box增加噪声，来产生proposals
        输入参数即为该帧目标的bounding box，而输出为产生的proposals以及每一个proposal与原box对应的iou(用于计算与IoUNet预测的iou的loss)
        '''
    def __call__(self, data: TensorDict):
        '''
        输入参数：
        	TensorDict类型的data，其必须包含'train_images'，'test_images'，'train_anno'，'test_anno'
        输出：
        	追加了'test_proposals'以及'proposal_iou'的TensorDict
        主要流程：
        1、对训练&测试数据，训练&测试边界框进行一次联合变换(joint transform)，这会调用Transform类的__call__()函数
        2、分别对train以及test数据进行边界框抖动、抖动裁剪以及指定的transform
        	2.1 对目标bounding box进行抖动，抖动的理由在开头的注释中有说明，会调用自身函数_get_jittered_box()
        	2.2 以这些抖动后的边界框为中心，对搜索区域进行裁剪
        	2.3 将这些输入到设置好的Transform中，这也会调用Transform类的__call__()函数
        3、对test数据box进行操作，产生proposals，这会调用类自身的_generate_proposals()函数
        4、封装为新的字典项后将其值返回
        '''
```

那么关于它的父类还是要说一说

```python
# ./ltr/data/processing.py
# 注释说明这是一个进行数据处理的基类
class BaseProcessing:
    def __init__(
        self, transform=transforms.ToTensor(), 
        train_transform=None, test_transform=None, 
        joint_transform=None):
        '''
        参数有四，分别是常规的transform，针对训练的transform，针对测试的transform，联合的transform
        最后一个应该是给边界框以及掩膜用的
        会用这些参数生成一个字典
        '''
    def __call__(self, data: TensorDict):
        # 需要继承的子类实现的方法，在直接使用该类实例的时候就会调用该方法，输入为TensorDict类型的参数
```

那么产生的应该是一个可以被调用的方法，就是data_processing_train

### 4.3 对验证数据进行处理

```python
data_processing_val = processing.ATOMProcessing(
    search_area_factor=settings.search_area_factor,
    output_sz=settings.output_sz,
    center_jitter_factor=settings.center_jitter_factor,
    scale_jitter_factor=settings.scale_jitter_factor,
    mode='sequence',
    proposal_params=proposal_params,
    transform=transform_val,
    joint_transform=transform_joint
)
```

其相同地使用了ATOMProcessing类，只是替换了其中的一个transform的参数

## 五、训练流程—采样器&加载器
### 5.1 训练数据采样器&加载器

```python
# The sampler for training
dataset_train = sampler.ATOMSampler(
    [lasot_train, got10k_train, trackingnet_train, coco_train],
    [1,1,1,1],
    samples_per_epoch=1000*settings.batch_size,
    max_gap=50, 
    processing=data_processing_train
)
# The loader for training
loader_train = LTRLoader(
    'train', dataset_train, training=True, 
    batch_size=settings.batch_size, 
    num_workers=settings.num_workers,
    shuffle=True, drop_last=True, stack_dim=1
)
```

看一眼采样器参数，第一个参数就是**3**中产生的图像数据，第五个参数则是**4.2**还有**4.3**里得到的训练数据的处理函数，这两个模块会在这里发挥作用

跟着调试进入函数内部，其结构如下

```python
# ./ltr/data/sampler.py
class ATOMSampler(TrackingSampler):
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames=1, num_train_frames=1, processing=no_processing, 
                 frame_sample_mode='interval'):
```

这个类只有一个init函数，感觉这个流程很像在设计Dataset，注意到它有一个父类**TrackingSampler**，而且注释也提示详情见这个父类，这里贴出父类的结构

```python
# ./ltr/data/sampler.py
'''
这是一个负责从训练集视频序列中采样帧形成batches的类
每一个训练样本为包含以下内容二元组：
1、一系列的训练帧
2、一系列的测试帧

采样的方法如下：
1、随机选择一个数据集(怪不得用4个数据集)
2、在这个数据集中选择一个视频序列，并采集一个base frame
3、在[base_frame_id - max_gap, base_frame_id]和(base_frame_id, base_frame_id + max_gap]中分别采样训练帧和测试帧，需要注意的是，只采样包含有目标的帧，如果采样的数量不足，会适当增大max_gap的数值，直到数量足够为止

采样完成之后的这些样本帧会输入到处理函数中(processing)进行必要的处理
'''
class TrackingSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, 
                 frame_sample_mode='causal'):
        '''
        参数含义
        datasets: 输入为用于训练的数据集的列表(说明可以是一个也可以是多个数据集)
        p_datasets: 表示每个数据集被抽样的概率(连这个都有准备)
        samples_per_epoch: 每一个epoch的训练样本数量
        max_gap: 采样的最大间距
        num_test_frames: 要采样的测试帧数量
        num_train_frames: 要采样的训练帧数量
        processing: 数据处理模块
        frame_sample_mode: 采样方式的选择，'causal'或者'interval'可选，随机抽样和间隔内随机抽样
        
        函数内容还是对这些参数的设置
        '''
    def __len__(self):
        # 返回samples_per_epoch这个参数的值
    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        '''
        在min_id和max_id之间采样num_ids个目标可视帧
        输入参数：
        	visible: 指明指定的帧中十分存在目标的一维张量
        	num_ids: 需要采样的帧的数量
        	min_id: 允许id下限
        	max_id: 允许id上限
        返回值：
        	帧的标签形成的列表
        '''
    def __getitem__(self, index):
        '''
        参数说明：
        	index：就是Dataset中__getitem__用的那个index
        返回值：
        	包含有所有data blocks的张量字典(TensorDict)
        主要流程如下：
        1.按照前面设定好的概率随机挑选一个数据集
        2.采样一个视频序列并保证其具有足够的目标可视帧
        	2.1 随机采样一个视频序列，会使用到该数据集的类的get_num_sequences()函数获取训练集视频数量，然后随机产生其中的一个id
        	2.2 统计该序列的目标可视帧的数量，需要用到数据集的get_sequence_info()函数，满足要求就返回该可视帧组成的列表visible，否则再回2.1
        3.会调用自己的_sample_visible_ids()函数进行采样，流程和开头说的相同，先找一个base_frame，分别采train和test，还有那个扩大搜索的机制
        4.调用数据集的get_frames()函数，获取视频详细信息
        5.按照以下封装返回值
        	TensorDict({'train_images': train_frames,
                       'train_anno': train_anno['bbox'],
                       'test_images': test_frames,
                       'test_anno': test_anno['bbox'],
                       'dataset': dataset.get_name(),
                       'test_class': meta_obj_test.get('object_class_name')})
        	同时注意到这里调用了self.processing(data)，也就是此处进行了Transform，再返回数据
        '''
```

该父类继承自**torch.utils.data.Dataset**，并且对len以及getitem两个函数进行了实现，之前也提过，这两个函数是自定义Dataset需要实现的，在这里实现的情况下，可以判断这个类的返回值也是一种Dataset的类型，最终可以看到其返回的信息量较多，且同时包含了训练样本和测试样本

采样器得到的结果会作为加载器的输入，经典的Dataset用于Dataloader

```python
# ./ltr/data/loader.py
class LTRLoader(torch.utils.data.dataloader.DataLoader):
    __initialized = False
    # 其仅有一个init函数，具体作用不明
    def __init__(self, name, dataset, training=True, batch_size=1,
                 shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None,
                 stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        '''
        参数含义：
        dataset: 输入需要是一个Dataset类或是继承的类
        batch_size: 每次迭代获取的batch大小
        suffle: 是否打乱
        后续的参数了解即可，因为一般不会改动
        '''
```

该LTRLoader增加额外的参数是为了在训练流程中给出指示，因此它实质上没有对Dataloader这个父类进行更改

Dataloader本质没有变化，只需要留意在新的类中可以用于Dataloader的参数，就是**dataset_train**、**batch_size**、**shuffle=True**

因此需要重点注意的就是继承Dataset类实现的ATOMSampler

### 5.2 验证数据采样器&加载器

```python
# The sampler for validation
dataset_val = sampler.ATOMSampler(
    [got10k_val],
    [1], 
    samples_per_epoch=500*settings.batch_size, 
    max_gap=50,
    processing=data_processing_val
)
# The loader for validation
loader_val = LTRLoader(
    'val', dataset_val, training=False,
    batch_size=settings.batch_size, 
    num_workers=settings.num_workers,
    shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1
)
```

与**5.1**相同，只是参数更改，此处就不再放上代码

## 六、训练流程—网络模型
### 6.1 网络模型创建

```python
# Create network and actor
# Phase 1
net = atom_models.atom_resnet18(backbone_pretrained=True)
# Phase 2
objective = nn.MSELoss()
# Phase 3
actor = actors.AtomActor(net=net, objective=objective)
```

终于来到了网络结构部分，从论文中可以获悉，网络分成了两个部分，离线学习的IoU预测器，以及在线学习的分类器，而该部分只包括了IoUNet的部分

首先是<font color='red'><b>第一阶段</b></font>，网络创建，Debug下会定位到原先参数指定的```atom.py```中

```python
# ./ltr/models/bbreg/atom.py
def atom_resnet18(
    iou_input_dim=(256,256), 
    iou_inter_dim=(256,256), 
    backbone_pretrained=True
):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)
    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(
        pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim
    )
    net = ATOMnet(
        feature_extractor=backbone_net, bb_regressor=iou_predictor,
        bb_regressor_layer=['layer2', 'layer3'],extractor_grad=False
    )
    
    return net
```

通过调用该方法来创建网络，从注释来看又分成两部分，骨干网络以及IoU预测器是分开产生的，之后再拼合成一个网络，不过这里好像没有看到和分类器有关的部分

先来到骨干网络部分，论文中也说明采用的ResNet18作为骨干网络，debug引向了下面这个函数

```python
# ./ltr/models/backbone/resnet.py
def resnet18(output_layers=None, pretrained=False, **kwargs):
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
```

应该是一个比较容易看懂的resnet构筑方法，那么该函数又指向了类ResNet

```python
# ./ltr/models/backbone/resnet.py
class ResNet(Backbone):
    # 采用的是pytorch源码的ResNet组成方法，已确认与Pytorch提供的ResNet18源代码框架一致
```

再来是IoU预测器的网络，这个就直接引向了一个类

```python
# ./ltr/models/bbreg/atom_iou_net.py
# 注释说明为IoU预测的模块
class AtomIoUNet(nn.Module):
    def __init__(
        self, input_dim=(128,256), 
        pred_input_dim=(256,256), pred_inter_dim=(256,256)):
        '''
        对网络模块进行设置，同时初始化权重
        该IoU预测器需要将骨干网络的两个特征层作为输入，论文中提及是block3以及block4
        输入参数：
        	input_dim: 两个输入的骨干层的特征维度
        	pred_input_dim: 输入到预测网络中的维度
        	pred_inter_dim: 预测网络的中间维数
        从论文图中可以看出，其存在两个分支，一个分支是给模板图像使用(代码用_r的变量表示)，另一个分支给查询图像使用(变量用_t表示)
        对模板分支：
        	1、为来自block3的特征图设置一个卷积层[128>128, size=(3,3), stride=1, padding=1]
        	2、为这个卷积层的输出设置一个PrRoIPooling[size=(3,3), scale=1/8]
        	3、为该PrRoIPooling的输出设置一个卷积层[128>256, size=(3,3), stride=1, padding=0]，按照pooling输出为3*3，可知经过这个卷                 积层以后就会变成1*1，相当于变成了一个一维向量，同时还提升了通道数量
        	4、为来自block4的特征图设置一个卷积层[256>256, size=(3,3), stride=1, padding=1]
        	5、为这个卷积层的输出设置一个PrRoIPooling[size=(1,1), scale=1/16]
        	6、这个PrPoIPooling的输出已经是256*1*1了，然后它和block3产生的256*1*1的向量组合，输入到一个设置好的卷积层中
        	[256+256>256, size=(1,1), stride=1, padding=1]，降维行为，论文称为Modulation，这个应该是调制的一个环节
        	7、同样为上面这个(256+256)*1*1的向量组合还设置了一个卷积层[256+256>256, size=(1,1), stride=1, padding=0]，调制的另一个环节
        至此，模板分支的流程结束
        对查询分支：
        	1、为来自block3的特征图设置一个卷积层[128>256, size=(3,3), stride=1, padding=1]，通道数增加，特征图尺寸不变化
        	2、这个卷积层的输出会再接一个卷积层[256>256, size=(3,3), stride=1, padding=1]
        	3、这次卷积完的输出会设置一个PrRoIPooling[size=(5,5), scale=1/8]
        	4、来自block4的特征图经过两个卷积层[256>256, size=(3,3), stride=1, padding=1]及[256>256, size=(3,3), stride=1, padding=1]
        	5、同样为输出设置一个PrRoIPooling[size=(3,3), scale=1/16]
        	6、此处应该有一个和模板分支最终产生的调制向量进行通道相乘的过程(注意到大家维度都是256)，当然在初始化函数里肯定没有声明
        	7、那么这个处理过的特征图(block3和block4是分开的)分别输入到两个LinearBlock中(应该是要先摊平)，获得一维向量
        	8、最终两个一维向量再拼接送入到预测器(就是个全连接层)中，输出值为1个，应该就是预测的IoU
        至此查询分支流程结束
        这其中应该还有一些策略，但是init中只能看出这么多，因为只是声明了网络构成
        '''
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)
        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(3, 3, 1/8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1/8)
        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)

        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)

        self.prroi_pool4r = PrRoIPool2D(1, 1, 1/16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)
        self.fc34_3r = conv(256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv(256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)

        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

        self.iou_predictor = nn.Linear(pred_inter_dim[0]+pred_inter_dim[1], 1, bias=True)
        # 后面还有个权重初始化模块，这个初始化方法还是不错的，感觉是万用插件
    def forward(self, feat1, feat2, bb1, proposals2):
        '''
        注释说明该函数只用于训练(论文说明IoU-Net是离线训练)，在Tracking时调用独立的函数即可
        输入参数：
        	feat1: reference frames中获取的特征
        	feat2: test frames中获取的特征
        	bb1: reference frames的GTBox
        	proposals2: test frames产生的proposal boxes
        其主要流程：
        1、进行一些处理
        2、将reference frames的特征feat1连同其GTBox作为输入得到调制向量，这里会调用get_modulation()函数
        3、将test frames的特征输入得到iou_feat，要调用get_iou_feat()函数
        4、将调制向量做一些处理，同时对预测的proposals也做一些处理 [64*16*4 proposal]
        5、将调制向量、iou_feat以及预测的proposals作为输入，调用函数predict_iou()，得到iou预测[64*16 视频共64帧，一帧16个proposals]
        '''
    def predict_iou(self, modulation, feat, proposals):
        '''
        注释说明该函数的作用是对于给定的边界框(proposals)，预测其IoU
        输入参数：
        	modulation: 来自reference部分的调制向量(两个组成的列表)
        	feat: test部分获得的iou_feat
        	proposals: 预测的边界框(应该是有一组)
        流程和之前猜测的好像有些不一样，其先将reference的调制向量和iuo_feat进行通道相乘
        再将其送入到prpooling中，和论文的图有些区别，应该是没有什么冲突才会这么做的，可能是为了减少计算量
        获得的RoI特征图再经由卷积变为1*1的特征，最后两个block的特征拼接，作为输入输入到全连接层中，获得iou_pred(长什么样还不知道)
        '''
    def get_modulation(self, feat, bb):
        '''
        注释说明该函数作用是获得调制向量
        输入参数：
        	feat: 来自reference的骨干网络提取特征
        	bb: 对应的bounding box信息
        流程和初始化时猜测的相同，对block3和block4获取的特征，按照卷积、prpooling的顺序获得两组特征
        再经过一定处理，将两个特征变为1*1的尺寸，拼接为一个一维向量
        一维向量分别输入到两个全连接层中，获得两个调制向量
        '''
        return fc34_3_r, fc34_4_r
    def get_iou_feat(self, feat2):
        '''
        注释说明该函数的作用是获取用于iou预测的特征
        输入参数：
        	feat2: test frame经由骨干网络提取得到的特征，同样是两个block中得到的
        流程和猜测相同，经过两个卷积层获得特征
        '''
        return c3_t, c4_t
```

完成两个组件的创建之后，就是将这两个网络拼合成一个部分返回，指向ATOMnet

```python
# ./ltr/models/bbreg/atom.py
# 注释说明为ATOM的网络模块
class ATOMnet(nn.Module):
    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        '''
        输入参数：
        	feature_extractor: 骨干网络特征提取器
        	bb_regressor: IoU预测模块，指optimization-based box refine
        	bb_regressor_layer: 要输入到bb_regressor中的来自feature_extractor的网络层组成的列表
        	extractor_grad: 指示特征提取器是否需要梯度
        '''
    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        '''
        简要流程：
        1、首先清楚输入，train_imgs为支持图像，test_imgs为查询图像，train_bb相当于train_imgs的GTBox，test_proposals则是用于预测IoU的box
        2、train_imgs和test_imgs会先输入到骨干网络中提取特征train_feat和test_feat
        3、稍作处理之后，上面得到的特征会和box信息一起输入到IoU预测器中，得到一个IoU预测
        '''
    def extract_backbone_features(self, im, layers=None):
        # 先指定要获得哪些层的特征，再去获得这些特征
    def extract_features(self, im, layers):
        # 用来调用特征提取器网络获得指定层的特征
```

至此完成网络部分的创建，而后进入<font color='red'><b>第二阶段</b></font>，对损失函数进行设置，采用的是MSELoss，适合于回归任务的损失函数

然后<font color='red'><b>第三阶段</b></font>将网络和损失函数放入到一个actor中，其指向了AtomActor

```python
# ./ltr/actors/bbreg.py
# 注释说明这是用来进行IoU-Net训练的
class AtomActor(BaseActor):
    # 没有init函数的声明，直接使用父类的
    def __call__(self, data):
        '''
        call函数中实现的训练语句，返回loss以及一些信息
        主要流程：
        1.用设置好的网络进行前馈传播，返回预测值
        2.通过设置好的损失函数计算损失
        3.将loss作为函数返回值，还封装了一个字典state一起返回，好像没啥用
        '''
```

来看看它的父类

```python
# ./ltr/actors/base_actor.py
# 注释说明Actor类的作用就是调度数据进入网络进行训练
class BaseActor:
    def __init__(self, net, objective):
        # 将输入的net以及objective作为类内参数self.net以及self.objective
    def __call__(self, data: TensorDict):
        # 需要继承它的子类实现的函数，其作用是对训练流程进行响应，相当于forward()，返回值是loss以及detailed losses
    def to(self, device):
        # 就是根据device的取值将net设置为cuda()或是cpu()
        self.net.to(device)
    def train(self, mode=True):
        # 指示网络使用训练模式还是测试模式，之前看过model.eval()源码其实就是model.train(False)
        self.net.train(mode)
    def eval(self):
```

那么训练所需要的模型就创建好了

### 6.2 优化器设置

```python
# Optimizer
# Phase 1
optimizer = optim.Adam(actor.net.bb_regressor.parameters(), lr=1e-3)
# Phase 2
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
```

## 七、训练流程—训练启动
```python
# Create Trainer
trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
# Run Training(set fail_safe=False if you are debugging)
trainer.train(50, load_latest=True, fail_safe=True)
```

这个LTRTrainer应该就是个综合调度的模块，现有的参数全都汇集到这里了

```python
# ./ltr/trainers/ltr_trainer.py
class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        '''
        主要参数信息：
        	actor：用于网络训练的actor，也就是之前6.1生成的
        	loaders：提供数据的数据加载器，这里就是一个列表[loader_train, loader_val]
        	optimizer：优化器，在之前6.2声明的是Adam
        	settings：训练的一些设置
        	lr_scheduler：学习率列表，也是前面的6.2语句产生的
        完成参数设置后的流程如下：
        1.调用_set_default_settings()
        2.对tensorboard进行初始化(这块就是记录训练过程用的，应该没什么关系)
        3.数据移至GPU?这是什么不知道
        '''
    def _set_default_settings(self):
        # 进行一些默认设置
    def cycle_dataset(self, loader):
        '''
        注释说明该函数进行以此训练或是验证的循环
        主要流程如下：
        1.self.actor.train(loader.training)
        	1.1 loader.training是之前5.1和5.2声明该loader时输入的参数，指示是否处于训练，对于训练集而言为True，对验证集而言是False
        	1.2 self.actor.train()为作为参数的actor的内部函数，用于指示模型是进入train()还是eval()
        2.torch.set_grad_enabled(loader.training)
        	同样的，根据loader.training是True还是False来指示是否自动计算梯度，验证的时候就不需要
        3.self._init_timing()
        	调用自身的_init_timing()函数，用于计时及num_frames初始化
        4.for i, data in enumerate(loader, 1): 从索引1开始读取数据，指i=1
        	4.1 对于数据的出现在5.1中有说明
        	4.2 数据按照设置变为GPU或是CPU
        	4.3 给data这个字典增加新的元素'epoch'和'settings'，分别为self.epoch和self.settings
        	4.4 前馈过程，输入到self.actor中获得loss
        	4.5 如果当前处于训练状态，就组织一个标准的反向传播
        	4.6 会调用自身的_update_stats()对训练信息进行更新，同时调用_print_stats()进行展示，就是刚刚那个actor里得到的state
        		[train: 1, 1 / 1000] FPS: 0.1 (0.1)  ,  Loss/total: 0.50103  ,  Loss/iou: 0.50103
            '''
    def train_epoch(self):
        '''
        必须要实现的父类函数
        由其父类BaseTrainer的train()函数来到该方法中
        其流程如下：
        1.从self.loaders中遍历元素，按照输入来看其实就是训练数据集和验证数据集两个，也就是一轮中要过一遍训练集和一遍验证集
        	1.1 调用类自身的cycle_dataset函数，输入参数为刚刚得到的数据集迭代器
        2.调用自身函数self._stats_new_epoch(self)
        3.调用自身函数self._write_tensorboard(self)
        '''
    def _init_timing(self):
        # 用于计时，同时会对self.num_frames进行初始化
    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # 好像是和训练的信息记录有关的
    def _print_stats(self, i, loader, batch_size):
        # 在控制台上打印训练信息的
    def _stats_new_epoch(self):
        # 记录学习率
    def _write_tensorboard(self):
        # 这个是给tensorboard记录用的函数
```

它的父类如下

```python
# ./ltr/trainers/base_trainer.py
class BaseTrainer:
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        '''
        接收参数与LTRTrainer的init函数相同
        过程中会调用两个函数
        1.自身的update_settings()，用于更新训练器的设置
        2.作为其参数的actor的to()函数
        '''
    def update_settings(self, settings=None):
        '''
        更新训练器设置，在初始化时必须调用
        其主要功能就是对训练的工作空间以及文件存储地址进行更新和创建
        '''
    def train(self, max_epochs, load_latest=False, fail_safe=True):
        '''
        LTRTrainer并没有实现这个类，因此会直接调用父类的这个函数
        训练器入口函数，其主要参数：
        	max_epochs：设置训练的最大轮次
        	load_latest：指示是否从最近的轮次恢复训练
        	fail_safe：指示是否在程序发生错误时重启训练
        整个函数主导训练的进行，其流程如下：
        1.首先在外面设置了一个崩溃规避的机制，有一定的尝试次数num_tries，不论是达到最大轮次还是程序崩溃重启次数超过尝试次数，函数都会终止
        2.根据 load_latest 决定是否加载上一轮次的训练参数
        3.进入到机制内部，其以最大训练轮次为循环参数进行迭代
        	3.1 设置 self.epoch 为当前训练轮次
        	3.2 调用函数 self.train_epoch()，继续运行后会进入到LTRTrainer实现的该函数中
        '''
    def train_epoch(self):
        # 需要继承的子类实现
    def save_checkpoint(self):
        # 存储模型使用
    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        # 加载模型使用
```

声明完成就是直接进行训练了，回调用train()函数启动训练流程

## 八、测试流程

由<font color='red'><b>./pytracking/run_tracker.py</b></font>进入

```python
# ./pytracking/run_tracker.py
def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb',
                sequence=None, debug=0, threads=0, visdom_info=None):
    '''
    输入参数：
    	tracker_name: 要使用的跟踪器名称
    	tracker_param: 要使用的权重文件，一般为default
    	run_id: 指示进程号的，不填没关系
    	dataset_name: 要用于测试的数据集
    	sequence: 默认为None，指测试整个数据集，当其有值时，仅测试该数据集中指定编号的视频序列
    	debug: 指示debug等级，越高在测试时展示的信息越多
    	threads: 用于指示线程，默认为0
    	visdom_info: 用于结果的可视化，不会影响到整个过程
    '''
    visdom_info = {} if visdom_info is None else visdom_info
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]
    # Phase 1
    trackers = [Tracker(tracker_name, tracker_param, run_id)]
    # Phase 2
    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info)
```

### 8.1 Tracker

```python
# ./pytracking/run_tracker.py
from pytracking.evaluation import Tracker
# Phase 1
trackers = [Tracker(tracker_name, tracker_param, run_id)]
```

其所使用的类Tracker如下：

```python
# ./pytracking/evaluation/tracker.py
# 以评估和运行为目的的封装的跟踪器
class Tracker:
    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        '''
        输入参数：
        	name: 跟踪方法的名称，这里我们使用的是'atom'
        	parameter_name: 权重参数文件的名称，一般为'default'
        	run_id: 运行的id，不填使用默认即可
        	display_name: 指示结果图表的名称，与流程没有影响
        初始化的主要目的是设置一些参数
        同时与pytracking.tracker.atom这个文件夹产生联系，可以理解为从./pytracking/tracker/atom/atom.py引入ATOM这个类
        (self.tracker_class = ATOM)
        '''
    def _init_visdom(self, visdom_info, debug):
    def _visdom_ui_handler(self, data):
    def create_tracker(self, params):
        # 利用已经有的参数设置初始化一个跟踪器，self.tracker_class即初始化时得到的ATOM这个类
        # 那么就相当于实例化一个类对象，详细可见8.3
        tracker = self.tracker_class(params)
        return tracker
    def run_sequence(self, seq, visualization=None, debug=None, visdom_info=None, multiobj_mode=None):
        '''
        在该跟踪器上运行一个视频序列
        输入参数：
        	seq: 要运行的视频序列
        	visualization: 什么东西的可视化，一般使用默认的None
        	debug: debug等级
        	visdom_info: 可视化信息
        	multiodj_mode: 只有多目标跟踪的时候会用到
        流程：
        1、调用自身的get_parameters()函数获取参数设置，并修改一些现有参数
        2、调用自身的_init_visdom()函数对可视化工具进行初始化，这个没有什么影响，同样的还有init_visualization()函数的调用是相同的道理
        3、init_info = seq.init_info()获取视频序列的初始信息，就是提供了边界框信息的第一帧
        	3.1 init_info为一个字典，仅封装一个元素'init_bbox'，这是一个四个元素的列表，猜测是[x, y, w, h]
        4、由于是单目标跟踪，接下来可以直接跳到tracker = self.create_tracker(params)，调用类自身create_tracker()方法得到跟踪器
        5、将跟踪器、视频序列以及第一帧信息作为参数，调用_track_sequence()方法来获得跟踪结果
        6、将跟踪结果返回
        '''
    def _track_sequence(self, tracker, seq, init_info):
        '''
        输出中的每一个字段都是一个列表，其中会包含每帧的跟踪器预测
        输出：
        	output = {'target_bbox': [],
        			  'time': [],
        			  'segmentation': []}
        	对于单目标跟踪而言，target_bbox[i]是对第i帧的目标边界框预测
        	time[i]则是该帧处理的时间，segmentation[i]为第i帧预测的掩码
        流程：
        1、调用自身_read_image()函数获得第一帧图像
        	1.1 seq.frames[0]得到的是第一帧图像的地址
        2、计时start_time = time.time()
        3、调用跟踪器的方法tracker.initialize()，输入为第1步得到的起始帧数据以及第一帧的bbox，该部分见8.3
        4、循环开始正式对后续帧进行跟踪
        	4.1 从给出的视频序列的第2帧开始读取
        	4.2 首先依旧是调用_read_image()获得图像数据
        	4.3 然后调用tracker的track()函数对该帧的目标进行检测
        '''
    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
    def run_webcam(self, debug=None, visdom_info=None):
    def run_vot2020(self, debug=None, visdom_info=None):
    def run_vot(self, debug=None, visdom_info=None):
    def get_parameters(self):
        # 获取参数，这里将会使用类pytracking.parameter.atom.default，它的parameters()将会把一些参数设置返回
        params = param_module.parameters()
    def init_visualization(self):
    def visualize(self, image, state, segmentation=None):
    def reset_tracker(self):
    def press(self, event):
    def _read_image(self, image_file: str):
        # 就是用opencv读取图像并返回
```

### 8.2 run_dataset

```python
# ./pytracking/run_tracker.py
from pytracking.evaluation.running import run_dataset
# Phase 2
run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info)
```

其调用的具体函数如下：

```python
# ./pytracking/evaluation/running.py
'''
在一个数据集(或一个视频序列)上运行[trackers](tracker可以是一个，也可以是复数个组成的列表)
输入参数：
	dataset: 视频序列组成的列表，用以形成一个数据集
	trackers: 跟踪器组成的列表
	debug: 指示debug的级别
	threads: 运行时使用的线程数量，默认为0
	visdom_info: 用于可视化的一些信息，与流程没有太大关系
'''
def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None):
    # 一般线程设置为0，因此会顺序执行(sequential)，关注如下部分即可
    # 其意义是，对一个视频序列，分别使用列表中的跟踪器跟踪一次，再对下一个视频序列重复操作
    for seq in dataset:
        for tracker_info in trackers:
            run_sequence(seq, tracker_info, debug=debug, visdom_info=visdom_info)
```

该函数使用的**run_sequence**函数如下：

```python
# ./pytracking/evaluation/running.py
# 可以理解为执行单元，在一个视频序列上使用一个跟踪器进行跟踪
def run_sequence(seq: Sequence, tracker: Tracker, debug=False, visdom_info=None):
    # 前面主要是对结果文件的存在进行确认，如果结果文件已经存在会报错
    # 若没有问题，它将会调用tracker.run_sequence()，详情请见8.1
    output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
```

### 8.3 ATOM

这是对8.1内容的补充，8.1的Tracker类在初始化时引入了一个**ATOM**类，它的具体内容如下：

```python
# ./pytracking/tracker/atom/atom.py
class ATOM(BaseTracker):
    multiobj_mode = 'parallel'
    def initialize_features(self):
        # 对IoU-Net(ResNet18+IoU模块)进行初始化，主要是模型的加载与一些参数的设置
    def initialize(self, image, info: dict) -> dict:
        '''
        利用视频序列第一帧的信息对跟踪器进行初始化
        参考论文的Online Tracking Approach章节，能搞清楚大概
        在线的训练增加了一个分类器部分，用于进行前景背景分类
        其是一个两层的分类器结构：
        	第一层(w1)是一个size=(1,1)的卷积层，用于通道降维，debug的时候发现应该是self.projection_matrix[256,64,1,1]
        	第二层(w2)是一个size=(4,4)的卷积层，包含一个输出层，输出每个2D位置的分类置信，应该是self.filter[1,64,4,4]，输出是单通道
        然后初始化是围绕第一帧进行的，主要做的事是对第一帧进行多样的数据增强，得到30个初始的training samples
        然后就是对w1和w2进行初始化(优化)，此处优化过程未明，可能论文所知的后序只对w2优化也在这里进行
        优化器的部分看着有点复杂，直接就跳过了
        还需要对第一帧得到的这些特征以及标签进行保存，这样后面进行跟踪时就不用再重复生成
        
        该函数有如下的主要流程：
        1、首先调用自身的self.initialize_features()，对在线需要用到的IoU-Net框架进行初始化与模型加载
        2、计算第一帧ground truth box的中心点以及边界框大小
        3、调整搜索区域(target_sz)
        	3.1 首先使用某种策略(torch.prod作用不明)计算当前target_sz所能达到的搜索区域大小
        	3.2 该算法对搜索区域存在一个接受的区间[min, max]，搜索区域的大小要满足在这个区间之中
        	3.3 如果搜索区域没能达到要求，算法会计算一个尺度因子target_scale，对target_sz做出修改，简单来说是等比例缩放，使得搜索区域满足要求
        	3.4 如果原本的搜索区域达到要求，target_scale的值就是1.0
        	3.5 最终target_sz变为base_target_sz
        4、确认是否使用IoU-Net，使用一个指示参数表示
        5、设置搜索区域
        	5.1 将搜索区域(base_target_sz计算得到)转换为一个正方形(square)，当作样本采样尺寸，其值是(288, 288)的box
        	5.2 上述步骤得到的经过卷积后所需要的大小，还需要换算成卷积前的大小，虽然还是(288, 288)，主注意到最大步长为16
        6、进行一些尺寸的设置
        7、对优化器的参数进行设置
        8、调用自身的self.init_learning()
        9、对图像进行变化(numpy_to_torch)
        10、调用自身的self.generate_init_samples()，返回值size为[30,256,18,18]
        11、调用自身的self.init_iou_net()，完成对训练分支的调制向量提取
        12、调用self.init_projection_matrix(x)，这是x也就是产生的初始样本第一次被使用，运行完后生成一个projection_matrix [64, 256, 1, 1]
        13、调用self.preprocess_sample(x)用于获取training sample，前后没有变化，产生的值赋予train_x
        14、调用self.init_label_function(train_x)，产生对应每个sample的前景背景分类信息，作为标签，init_y
        15、调用self.init_memory(train_x)
        	这个有点意义不明，最终保存一个[250, 64, 18, 18]的self.training_samples，细看维度和projection_matrix的输出有点像
        16、调用self.init_optimization(train_x, init_y)，对分类器进行优化
        17、最后保存一下得到的目标中心，将结果返回
        '''
    def init_optimization(self, train_x, init_y):
        '''
        输入为此前以第一帧为模板生成的30个初始样本train_x [30, 256, 18, 18]，以及它们对应的目标状态标签init_y [30, 1, 18, 18]
        1、首先产生一个[1, 64, 4, 4]的self.filter，这个其实就是分类器的第二个部分w2，是一个卷积层
        2、然后按照某种初始化策略(randn)对这个卷积层进行初始化
        3、优化器设置为GaussNewtonCG，应该是论文中提到的其自己设计的梯度下降方法
        4、设置因式联合优化(以一个if语句为标识)：
        	4.1 函数会运用当前可用的参数创建一个FactorizedConvProblem实例self.joint_problem
        		需要注意的是self.init_training_samples [30, 256, 18, 18]和init_y成为了它的参数
        	4.2 self.projection_matrix (w1)与self.filter (w2)相连接为一个joint_var
        	4.3 对优化器进行初始化，首先我们知道当前的优化器是GaussNewtonCG，算法将会生成一个GaussNewtonCG实例self.joint_optimizer
        		需要注意的是self.joint_problem和joint_var均被当作了它的参数
        	4.4 进行一次联合优化，将会调用self.joint_optimizer.run()方法
        5、调用self.project_sample(self.init_training_samples, self.projection_matrix)对样本进行一次重新映射
           生成compressed_samples [30, 64, 18, 18]，确实像是经过了w1卷积层降维的样子
        '''
    def track(self, image, info: dict = None) -> dict:
        '''
        对给定的图像进行目标的跟踪(检测)，返回的值就是一个封装了目标边界框信息的字典
        参考论文的Online Tracking Approach章节了解
        首先根据上一帧的跟踪结果，用上一帧的box尺寸以及一个scale参数在这一帧中获得特征图搜索区域
        然后拿classification进行前景背景分类，得到一个分类置信图，取得分最高的2D坐标，以其为中心和上一帧的box尺寸组成新的box
        还需要对这个box进行追加噪声抖动，生成10个proposal，送入到IoU模块中
        取IoU预测最高的前三个proposal进行融合，最后得到这一帧的目标边界框
        同时还要对分类器和scale进行更新
        
        1、获取样本
        	1.1 获得上一帧的目标中心sample_pos以及sample_scales
        	1.2 调用self.extract_processed_sample()得到已经经过w1的样本特征图 [1, 64, 18, 18]，记为test_x
        2、计算分类置信得分
        	2.1 调用自身self.apply_filter(test_x)，将上一步得到的特征图输入到w2中，得到[1, 1, 18, 18]的分类置信scores_raw
        	2.2 调用self.localize_target(scores_raw)，将会返回 translation_vec, scale_ind, s, flag
        3、更新位置信息以及尺度信息
        	3.1 通过self.update_state()函数更新目标中心点self.pos，但是对当前的sample_pos没有影响
        	3.2 调用self.refine_target_box()，完成该帧边界框预测
        4、更新memory，使用self.update_memory(train_x, train_y, learning_rate)，存入本帧以及对应的生成标签
        5、按照一定策略会对分类器的w2部分进行再训练
        6、最后将这一帧的预测信息返回
        '''
    def apply_filter(self, sample_x: TensorList):
        # 将特征图输入到分类器的w2中(也就是self.filter)得到分类置信
    def localize_target(self, scores_raw):
        '''
        将得到的分类置信特征图[18, 18]，上采样为[288, 288]，也就是和样本同样大小
        然后会调用self.localize_advanced(scores) (scores: [288, 288])
        '''
    def localize_advanced(self, scores):
        # 会计算一个偏移，和上采样产生的误差有关
    def extract_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        # 将输入的图像，按照目标中心，切割下一块尺寸为sz的图像，并将其输入到骨干网络中，获得block4的特征图
        # 因此其输出为[1, 256, 18, 18]的特征图
    def get_iou_features(self):
    def get_iou_backbone_features(self):
    def extract_processed_sample(self, im: torch.Tensor, pos: torch.Tensor, 
                                 scales, sz: torch.Tensor) -> (TensorList, TensorList):
        # 依次对self.extract_sample()、self.project_sample()以及self.preprocess_sample()三个函数进行调用
    def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
        # 在ATOM的设置中相当于一个恒等变化
    def project_sample(self, x: TensorList, proj_matrix = None):
        # 将特征图输入到分类器的w1部分中(也就是self.projection_matrix)，得到映射过后的特征图
    def init_learning(self):
        # 依旧是对一些训练参数进行设置
        # 对分类器的两个层的激活函数进行设置
    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        '''
        产生经过增强的初始样本
        对第一帧进行多样的数据增强，产生30个初始的training samples
        返回值是这些样本的特征图 [30,256,18,18]
        
        需要重点注意的函数是下面这个
        self.params.features.extract_transformed(im, self.pos, self.target_scale, aug_expansion_sz, self.transforms)
        该函数会进行图像crop，应用图像变换，最终将产生的patch(数量为transforms的元素数量)送入到ResNet18中，将会返回block4得到的特征图
        会发现这些patch的数量不满足于论文所说的30个，这里是23个
        剩下的7个，代码补足的方式是复制7个同样的transform方法(第一个)，将其扩充到transforms列表的末尾
        同时对复制的方法对应的patch的特征图(也就是第一个patch)，将其也复制7份，追加一个dropout层，这样就能得到“相同”但是不同的7个特征
        同样追加到样本的末尾，那么就产生了30个样本
        '''
    def init_projection_matrix(self, x):
        # 生成分类器第一层w1，并对w1进行初始化
    def init_label_function(self, train_x):
        '''
        输入的数据train_x是在前面的步骤中产生的30个初始样本 [30, 256, 18, 18]
        首先函数会根据算法中的设置，生成一个零矩阵的列表，记为y，这个y的规格是 [250, 1, 18, 18]，用于存放真实标签
        对比train_x和y可以发现，他们在2、3两维是相同的，均为18*18，那么y中一个元素的实际规格就是[1, 18, 18]
        可以用来指示样本中的目标情况，也就是指明哪里存在目标
        而y的可容纳元素数量为250，输入的样本数量为30，应该是对最大容纳数量的设置
        再来对于指示目标的方式，是给[18, 18]的每个元素赋予一个[0, 1]的值，那么其值越高，说明其越接近于目标
        而这个产生的方式，由于输入的样本是可以知晓目标中心的位置(边界框中心)，而算法用使用了某个策略(不细究了)，利用这个中心点补全整个[18, 18]
        于是就形成了一个样本的标签，对所有输入数据重复这样的行为，就完成了train_x的对应标签的初始化，返回y中含有标签的元素[30, 1, 18, 18]
        '''
    def init_memory(self, train_x):
        # 后续更改企划
        '''
        1、事前准备
        	self.num_init_samples记录当前的样本数量，也就是初始的30个样本
        	self.init_sample_weights计算一个初始的采样权重(一个值)
        	self.init_training_samples保存初始训练样本
        2、生成计数器和采样权重数组，将计算得到的self.init_sample_weights赋值到self.sample_weights中(长度为250)
        3、初始化存储区域，将初始样本存入到self.training_samples中 [250, 64, 18, 18]，以后的调度也是从self.training_samples进行
        '''
    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate = None):
        # 后续更改企划
        # 这个部分比较奇特，暂时没法确认存储区在充满后的更新策略，后续需要时再准备
        # 项目暂时到此位置
    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, 
                              num_init_samples, fparams, learning_rate = None):
    def get_label_function(self, sample_pos, sample_scale):
    def update_state(self, new_pos, new_scale = None):
    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        '''
        原图被crop成了(288, 288)的包含有目标对象边界框的patch
        因此在这个(288, 288)的patch中的bbox的位置和尺寸信息也需要重新计算(需要在这个新的坐标系下)
        具体算法还没搞明白，感觉是这个意思
        '''
        box_center = (pos - sample_pos) / sample_scale + (self.iou_img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])
    
    def init_iou_net(self):
        '''
        这个有比较需要注意的地方，该函数不仅仅初始化了IoU-Net
        函数的主要流程：
        1、加载IoU-Net配置
        	1.1 加载IoU-Net的IoU网络部分(iou_predictor)
        	1.2 对iou_predictor中的所有参数设置requires_grad = False，即不进行梯度计算，也就是在线跟踪阶段不做更新
        2、调用自身self.get_iounet_box()，获得要输入到IoU-Net中的标准bbox信息
        3、获取了bbox信息后，对于做了transform的30个样本，bbox也要进行对应的变换，得到30个变换后的box信息，存于target_boxes中
        3、后面是按照IoU-Net的training branch流程，最终得到两个[256,1,1]的调制向量
        '''
    def refine_target_box(self, sample_pos, sample_scale, scale_ind, update_scale = True):
        '''
        主要流程如下：
        1、调用函数self.get_iounet_box()，计算正确的边界框坐标，返回值为4元素向量
        2、调用self.get_iou_features()，将sample输入到骨干网络中，得到ResNet18的bloc3&4的特征图
        3、通过抖动在初始边界框的基础设生成10个候补边界框init_boxes
        4、将2&3步得到的特征以及proposals作为输入，调用self.optimize_boxes(iou_features, init_boxes)
        5、对得到的output_boxes以及output_iou做一些处理
        6、选择top k个边界框候选进行融合，得到最终的目标边界框
        7、将本次的边界框相关信息保存
        '''
    def optimize_boxes(self, iou_features, init_boxes):
        '''
        对输入的boxes进行优化，使它们能在有限步骤内达到最大的iou
        最终输出output_boxes [10, 4], output_iou [10, 1]
        '''
```

而它的父类**BaseTracker**为它提供了init函数：

```python
# ./pytracking/tracker/base/basetracker.py
class BaseTracker:
    def __init__(self, params):
        self.params = params
        self.visdom = None
    def predicts_segmentation_mask(self):
    def initialize(self, image, info: dict) -> dict:
    def track(self, image, info: dict = None) -> dict:
    def visdom_draw_tracking(self, image, box, segmentation=None):
```

### 8.4 FactorizedConvProblem

```python
# ./pytracking/tracker/atom/optim.py
class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y:TensorList, 
                 filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation):
        # 输入主要是训练数据[256, 18, 18]以及其对应标签[1, 18, 18]，还有分类器的两个激活函数
    def __call__(self, x: TensorList):
        '''
        用于计算残差，输入参数x的组成应该是[filter, projection matrices]，也就是[w2, w1]
        输出为[data_terms, filter_regularizations, proj_mat_regularizations]
        步骤如下，具体的部分请看下面写的解说：
        	1、从x中分割出w2和w1，分别赋予filter和p
        	2、将训练数据输入到w1中，并经过w1的激活函数，得到输出
        	3、将从2得到的卷积结果再输入到w2中，经过w2的激活函数，得到输出(residuals)
        	4、将其与真实标签y相减得到残差第一部分(未完成)
        	5、这个第一部分还需要乘以对应的权重
        	6、然后计算网络的w1和w2，并配上对应的权重，得到残差的第二部分
        	7、两个部分组合，得到残差r(w)，作为结果返回
        '''
    def ip_input(self, a: TensorList, b: TensorList):
    def M1(self, x: TensorList):
```

这个类第一次出现在**ATOM**(8.3)类的方法**init_optimization**中，用于生成变量**self.joint_problem**

这个部分其实是论文的优化问题的实现，在此记录一下论文的内容以便参考

首先关于模型，论文表明其目标分类模块是一个两层的神经网络结构，其可以建模为
$$
f(x;w)=\phi_{2}(w_{2}*\phi_{1}(w_{1}*x))\tag{1}
$$
其中，${x}$就是经由骨干网络得到的特征图，$w=\{w_{1}, w_{2}\}$是分类器网络参数，${\phi_{1}}$和${\phi_{2}}$分别是这两层的激活函数，而${*}$就是卷积操作

那么作者受到DCF的启发，因此设计了一个相似的损失函数
$$
L(w)=\sum^{m}_{j=1}\gamma_{j}||f(x_{j};w)-y_{j}||^{2}+\sum_{k}\lambda_{k}||w_{k}||^{2}\tag{2}
$$
那么${y_{j}}$就是对应的特征图${x_{j}}$的真实标签，就是之前由label_function生成的分类置信图，每一个训练样本的影响程度都由权重参数${\gamma_{j}}$来控制，增加的权重的正则化项的影响程度也由${\lambda_{k}}$来控制

为了完成在线的优化，作者将公式(2)的问题打造成了一个残差的问题，这也是上面的算法步骤的由来

作者首先将损失函数第一项定义为了${r_{j}(w)=\sqrt{\gamma_{j}}(f(x_{j};w)-y_{j})}$，其中${j\in\{ 1,...m \}}$

同时，作者将第二项定义为了${ r_{m+k}(w)=\sqrt{\lambda_{k}}w_{k} }$，其中${ k=1,2 }$

这样以来，以上两者进行组合，损失函数就可以变成
$$
L(w)=||r(w)||^{2}
$$
那么上面的**call**函数的步骤就是按照论文的定义计算${r(w)}$过程

这个后续的变成共轭梯度问题的部分有些困难就不再多叙述了，可以参考论文中列出的算法

### 8.5 GaussNewtonCG

```python
# ./pytracking/libs/optimization.py
class GaussNewtonCG(ConjugateGradientBase):
    def __init__(self, problem: L2Problem, variable: TensorList, cg_eps = 0.0, fletcher_reeves = True,
                 standard_alpha = True, direction_forget_factor = 0, debug = False, analyze = False, plotting = False,
                 visdom=None):
        '''
        从初始化函数可以发现比较重要的是problem以及variable这两个参数，其他的都是指示性的参数
        problem输入的是FactorizedConvProblem类的实例，而variable则是分类器的网络参数
        以上两个参数分别赋予self.problem与self.x
        '''
    def clear_temp(self):
    def run_GN(self, *args, **kwargs):
    def run(self, num_cg_iter, num_gn_iter=None):
        '''
        该方法运作的入口函数，用来运行优化器
        其输入的含义：
        	num_cg_iter 每一次高斯牛顿法内部所进行的共轭梯度法的迭代次数，和论文所说一样，第一帧给到了10次
        	num_gn_iter 高斯牛顿法的迭代次数，第一帧给到了6次
        此外本函数主要的步骤就是组织外循环
        
        for cg_iter in num_cg_iter:
            self.run_GN_iter(cg_iter)
        
        此处的cg_iter均为列表[10]，且迭代进行6次(高斯牛顿法外循环6次)
        '''
    def run_GN_iter(self, num_cg_iter):
        '''
        进行一次高斯牛顿法的迭代，和论文的算法对照更清楚
        步骤如下：
        	1、对分类器网络self.x中所有参数requires_grad_设置为True，也就是w1和w2均参与更新
        	2、将self.x作为参数输入到self.problem中，这将会调用FactorizedConvProblem的__call__函数，得到初始残差self.f0
        	3、参数的准备工作
        	4、调用函数self.run_CG()，这是继承自父类的方法
        '''
    def A(self, x):
    def ip(self, a, b):
    def M1(self, x):
    def M2(self, x):
    def evaluate_CG_iteration(self, delta_x):
```

第一次出现也是在**ATOM**类的方法**init_optimization**中，它将**FactorizedConvProblem**的实例和分类器网络**joint_var**作为参数输入

这个是用来推导论文中的高斯牛顿共轭梯度法对分类器网络进行快速优化

它的父类如下

```python
# ./pytracking/libs/optimization.py
class ConjugateGradientBase:
    def __init__(self, fletcher_reeves = True, standard_alpha = True, direction_forget_factor = 0, debug = False):
    def reset_state(self):
    def run_CG(self, num_iter, x=None, eps=0.0):
        '''
        推导共轭梯度下降法的算法，结合论文中的算法看的话会比较清楚
        代码中的变量并不是按照论文算法命名的因此对照起来有点困难
        '''
    def A(self, x):
    def ip(self, a, b):
    def residual_norm(self, r):
    def check_zero(self, s, eps = 0.0):
    def M1(self, x):
    def M2(self, x):
    def evaluate_CG_iteration(self, x):
```







