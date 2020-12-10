# 基于PaddleHub的中秋节看图写诗

此生此夜不长好，明月明年何处看。万里乾坤双鬓改，一身风露寸心寒。

# 一、项目背景

中秋节以月之圆兆人之团圆，为寄托思念故乡，思念亲人之情，祈盼丰收、幸福。如今，在中秋节赏月、吃月饼、猜灯谜等已经成为了这一天必不可少的项目。

古有文人墨客举杯邀明月，今有百度飞桨看月写诗。为了将我们中华民族的优秀传统文化发扬光大，我也打算写一写诗，但不同的是，我打算让AI自己写诗。

还记得上一个版本的看图写诗吗？我已经将看图写诗通用版打包成了一个module，大家可以在PaddleHub上直接使用了：
- [reading_pictures_writing_poems](https://www.paddlepaddle.org.cn/hubdetail?name=reading_pictures_writing_poems&en_category=TextGeneration)

这是一个通用版本，但是对于一些特定场景来说，实现效果并不理想。

为了让大家能有更好的体验，我打算将看图写诗升级一下，让模型写出的诗更丰富且将中秋节的元素包含进去，做一个中秋节看图写诗的特辑。

# 二、实现思路

## 通用版看图写诗思路

一个月前，我已经实现了看图写诗的功能，这是一个通用版本：
- [还在担心发朋友圈没文案？快来试试看图写诗吧！](https://aistudio.baidu.com/aistudio/projectdetail/738634)

我们回顾一下通用版本看图写诗的思路：
1. 获取图像的分类信息，即知道该图片是什么
2. 根据分类信息拓展关键词并将若干关键词组合成古诗上阕
3. 将古诗的上阕送入古诗生成模型，获取完整的古诗词

## 中秋节特辑版看图写诗思路

其实主要思路是不变的，但是为了让模型效果更好，写出的古诗更多样性，我做了调整，改进后的思路如下：
1. 使用目标检测获取图像内物体的信息，具体识别对象为月亮和月饼
2. 将识别出来的对象名称送入自己训练的文本生成模型，得到古诗的上阕
3. 将古诗的上阕送入ernie_gen_poetry古诗生成模型中，得到完整古诗

## 两个版本的区别

主要的区别就在第二步：
- 获取图像信息后，怎么得到完整的古诗

第一个版本的做法是将关键组合起来，毫无逻辑地组合成古诗的上阕；<br>
第二个版本的改进就在于我做了一个根据关键词生成古诗上阕的模型。

# 三、具体实现

原来我直接调用现成的图像分类以及古诗生成模型实现的通用版看图写诗。为了让中秋特辑版的看图写诗效果更好，我使用自己训练的目标检测模型以及古诗生成模型，因此，下面将按目标检测和古诗生成的步骤展开。

因为我这里用了迁移学习，所以需要自建数据集。为了把这一过程讲清楚，我把具体实现分为**数据集准备**和**模型训练**两个步骤。

## 1.目标检测——获取图像信息

我使用PaddleX做目标检测，检测图像内的**月亮、月饼、灯笼和兔子**，因为这些都跟中秋节息息相关。

### 数据集准备

使用PaddleX训练模型时，不需要代码，但是数据集一定要高质量。

#### 数据集格式

在目标检测任务中，需要选定数据集所在文件夹路径(路径中仅含一个数据集) 。不支持.zip、tar.gz等压缩包形式的数据导入。

图片格式支持png, jpg, jpeg, bmp格式;标签格式检测数据集为.xml

- 图片文件夹命名需要为"JPEGImages"；
- 标签文件夹命名需要为”Annotations"

#### 数据集样例

![https://ai-studio-static-online.cdn.bcebos.com/e49fe031731f4f5d91d22693045ff6d616512600676e4ed6bdf264a4ab85ca36](https://ai-studio-static-online.cdn.bcebos.com/e49fe031731f4f5d91d22693045ff6d616512600676e4ed6bdf264a4ab85ca36)


### 模型训练

PaddleX的模型训练过程非常简单，只要数据集格式正确，基本没有问题：

 <img src="https://ai-studio-static-online.cdn.bcebos.com/6f12e438f112490288c2e490527781c58738c3eaa760449faddbe5eaa9ca0481" width="60%" height="60%">

训练完成后，直接导出模型即可。我已将导出的模型上传至本项目中，位于inference_model目录下。


### 调用目标检测模型

#### 安装paddlex

脚本运行依赖paddlex，这里我们用pip安装一下：



```python
!pip install paddlex
```

#### 获取图片内信息

以下代码是PaddleX官方提供的predict代码，能看到输入图片的预测结果：


```python
import paddlex as pdx

# 模型加载, 请将path_to_model替换为你的模型导出路径
# 可使用 mode = pdx.load_model('path_to_model') 加载
# 而使用Predictor方式加载模型，会对模型计算图进行优化，预测速度会更快
print("Loading model...")
model = pdx.deploy.Predictor('inference_model', use_gpu=True)
print("Model loaded.")

# 模型预测, 可以将图片替换为你需要替换的图片地址
# 使用Predictor时，刚开始速度会比较慢，参考此issue
# https://github.com/PaddlePaddle/PaddleX/issues/116
result = model.predict('lantern.jpg')

print(result)

# 可视化结果, 对于检测、实例分割务进行可视化
if model.model_type == "detector":
    # threshold用于过滤低置信度目标框
    # 可视化结果保存在当前目录
    pdx.det.visualize('lantern.jpg', result, threshold=0.3, save_dir='./')

```

    2020-10-19 19:13:38,273-INFO: font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
    2020-10-19 19:13:38,627-INFO: generated new fontManager


    Loading model...
    Model loaded.
    [{'category_id': 0, 'bbox': [183.08929443359375, 7.7965850830078125, 250.32818603515625, 311.2488250732422], 'score': 0.013529560528695583, 'category': 'MoonCake'}, {'category_id': 1, 'bbox': [189.15859985351562, 46.0416259765625, 235.4840087890625, 336.6524353027344], 'score': 0.9346879720687866, 'category': 'lantern'}, {'category_id': 2, 'bbox': [181.138916015625, 29.884109497070312, 255.14971923828125, 317.12022399902344], 'score': 0.4133519232273102, 'category': 'moon'}, {'category_id': 3, 'bbox': [183.08929443359375, 7.7965850830078125, 250.32818603515625, 311.2488250732422], 'score': 0.020459380000829697, 'category': 'rabbit'}]


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):


    2020-10-19 19:13:42 [INFO]	The visualized result is saved as ./visualize_lantern.jpg


#### 选择得分最高的检测结果


```python
best = {'score':0, 'category':'none'}
for item in result:
    if (item['score'] > best['score']):
        best['score'], best['category'] = item['score'], item['category']

print(best)
```

    {'score': 0.9346879720687866, 'category': 'lantern'}


#### 将检测结果转换成中文关键字


```python
if best['category'] == 'MoonCake':
    objects = ['月饼']
elif best['category'] == 'moon':
    objects = ['月亮']
elif best['category'] == 'lantern':
    objects = ['灯笼']
elif best['category'] == 'rabbit':
    objects = ['兔子']
else:
    objects = ['中秋节']
```

## 2.古诗生成——根据关键词直接生成古诗

### 数据集准备

在文本生成任务中，数据集的好坏决定了模型最后的效果，因此这里我把数据集单独拿出来讲解。

#### 数据集格式

我们在自建数据集的时候，一定要确保数据集的格式正确，否则在训练过程中就会出现报错的情况。

训练集的格式应为：
- "序号\t输入文本\t标签"

验证集的格式应为：
- "序号\t输入文本\t标签"

这里需要注意的是，\t是跳格，虽然看上去能用4个空格代替，但实际上如果用4个空格的话，会导致训练失败。

#### 数据集样例

训练集样例：

> 1	月亮	万里无云镜九州，最团圆夜是中秋。<br>
2	月亮	一年逢好夜，万里见明时。<br>
3	月亮	三五夜中新月色，二千里外故人心。<br>
4	月亮	良夜清秋半，空庭皓月圆。<br>
5	月饼	黄白翻毛制造精，中秋送礼遍都城。<br>
6	月饼	论斤成套多低货，馅少皮干大半生。<br>
7	月亮	世远月何在，天空人自圆。<br>
8	月亮	此夜明月夜，清光十万家。<br>
9	月亮	举头望圆月，低头思故乡。<br>
10	月亮	今夜月明人尽望，不知秋思落谁家。<br>

完整数据集我已放到work目录下，大家可以自行查看：

 <img src="https://ai-studio-static-online.cdn.bcebos.com/83e1d2f2a20c45c5b9fb1d9bf78a01890b530a567ab24936bb8bc3f8f2bccbf5" width="70%" height="70%">



### 模型训练

这里我将在ernie-gen的基础上做finetune。

#### 1.将PaddleHub更新到最新版本

ernie-gen需要：
- paddlepaddle >= 1.8.2
- paddlehub >= 1.7.0

因此这里直接把PaddleHub更新到最新版本：


```python
!pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
```

  

#### 2.安装ERNIE-GEN

ERNIE-GEN 是面向生成任务的预训练-微调框架，我们在使用下面这两条命令安装一下：


```python
!hub install ernie_gen==1.0.1
!pip install paddle-ernie
```


#### 3.在ERNIE-GEN的基础上做finetune

在ERNIE-GEN的基础上做finetune有专门的微调API，具体参数如下：

- train_path(str): 训练集路径。训练集的格式应为："序号\t输入文本\t标签"，例如："1\t床前明月光\t疑是地上霜"
- dev_path(str): 验证集路径。验证集的格式应为："序号\t输入文本\t标签"，例如："1\t举头望明月\t低头思故乡"
- save_dir(str): 模型保存以及验证集预测输出路径。
- init_ckpt_path(str): 模型初始化加载路径，可实现增量训练。
- use_gpu(bool): 是否使用GPU。
- max_steps(int): 最大训练步数。
- batch_size(int): 训练时的batch大小。
- max_encode_len(int): 最长编码长度。
- max_decode_len(int): 最长解码长度。
- learning_rate(float): 学习率大小。
- warmup_proportion(float): 学习率warmup比例。
- weight_decay(float): 权值衰减大小。
- noise_prob(float): 噪声概率，详见ernie gen论文。
- label_smooth(float): 标签平滑权重。
- beam_width(int): 验证集预测时的beam大小。
- length_penalty(float): 验证集预测时的长度惩罚权重。
- log_interval(int): 训练时的日志打印间隔步数。
- save_interval(int): 训练时的模型保存间隔部署。验证集将在模型保存完毕后进行预测。


<br>
运行结果是一个字典，包含2个键:

- last_save_path(str): 训练结束时的模型保存路径。
- last_ppl(float): 训练结束时的模型困惑度。

具体代码请前往GitHub查看：

[https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen)


```python
import paddlehub as hub

module = hub.Module(name="ernie_gen")           # 加载预训练模型

result = module.finetune(
    train_path='/home/aistudio/work/train.txt', # 训练集路径
    dev_path='/home/aistudio/work/dev.txt',     # 验证集路径
    save_dir="ernie_gen_result",                # 模型保存及验证集输出路径
    max_steps=600,                              # 训练次数
    noise_prob=0.2,                             # 噪声概率
    batch_size=5,                               # 训练时batch的大小
    log_interval=10                             # 日志打印间隔
)

module.export(params_path=result['last_save_path'], module_name="MidAutumnPoetry", author="zbp") # 模型转换
```


#### 4.使用模型API预测

模型转换完毕之后，通过hub install $module_name安装该模型


```python
!hub install MidAutumnPoetry
```

    Successfully installed MidAutumnPoetry


通过API预测的方式调用自制module：


```python
import paddlehub as hub

module = hub.Module(name="MidAutumnPoetry")

# test_texts = objects
test_texts = ['月亮', '月饼', '灯笼', '兔子']

# generate包含3个参数，texts为输入文本列表，use_gpu指定是否使用gpu，beam_width指定beam search宽度。
results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
for result in results:
    print(result)
```

    [32m[2020-10-21 19:20:48,679] [    INFO] - Installing MidAutumnPoetry module[0m
    [32m[2020-10-21 19:20:48,807] [    INFO] - Module MidAutumnPoetry already installed in /home/aistudio/.paddlehub/modules/MidAutumnPoetry[0m
    [33m[2020-10-21 19:20:57,409] [ WARNING] - use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True[0m


    ['此生此夜不长好，明月明年何处看。', '山中夜来月，到晓不曾看。', '海上生明月，天涯共此时。', '举头望明月，低头思故乡。', '此生此夜不长好,明月明年何处看。']
    ['中秋鲜果列晶盘，饼样圆分桂魄寒。', '问天中秋何时到，美酒酥饼得真知。', '何须急管吹云暝，高寒滟滟开金饼。', '巧出圆月形，貌得婵娟月。', '每逢中秋赏明月，要同圆月作团圆。']
    ['梅花映雪挂灯笼，福字生金万户红。', '目穷淮海满如银，万道虹光育蚌珍。', '高梧月白绕飞鹊，灯笼蜡纸明空堂。', '暂得金吾夜，通看火树春。', '停车傍明月，走马入红尘。']
    ['吴质不眠倚桂树，露脚斜飞湿寒兔。', '玉兔香笺谁与共，黄昏送走又清晨。', '金乌玉兔来还去，一卷风云说到今。', '吴质不眠倚桂树，玉兔斜飞湿寒兔。', '定知玉兔白如月，化作霜风九月寒。']


#### 5.根据古诗的上阕生成古诗的下阕

安装生成古诗的模型ernie_gen_poetry


```python
!hub install ernie_gen_poetry==1.0.0 #生成古诗词
```

    Downloading ernie_gen_poetry
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmpgysxbhai/ernie_gen_poetry
    [==================================================] 100.00%
    Successfully installed ernie_gen_poetry-1.0.0


通过API预测的方式调用古诗生成模型：


```python
module = hub.Module(name="ernie_gen_poetry") # 调用古诗生成的模型

FirstPoetry = [results[0][0]] # 使用上阕生成下阕
SecondPoetry = module.generate(texts=FirstPoetry, use_gpu=True, beam_width=5)

Poetrys = []
Poetrys.append(FirstPoetry[0])
Poetrys.append(SecondPoetry[0][0])

print("生成的古诗词是：")
print("{}{}".format(Poetrys[0], Poetrys[1]))

```

    [32m[2020-10-21 19:22:27,761] [    INFO] - Installing ernie_gen_poetry module[0m
    [32m[2020-10-21 19:22:27,926] [    INFO] - Module ernie_gen_poetry already installed in /home/aistudio/.paddlehub/modules/ernie_gen_poetry[0m


# 四、模型贡献

PaddleHub的初衷是方便开发者们快速地调用深度学习模型，因此，我们可以把我们做好的模型制作成一个module贡献给PaddleHub的模型库。

## PaddleX模型转换成PaddleHub模型

PaddleX是一个图形化的深度学习训练工具，对于零基础的同学来说，也能快速地上手。因此，这里我选择了PaddleX。

但是PaddleX的模型无法直接供PaddleHub使用，所以首先需要把PaddleX的模型转成PaadleHub的模型，模型转换具体项目：
- [手把手带你将Paddlex模型部署为PaddleHub（作者：七年期限）](https://aistudio.baidu.com/aistudio/projectdetail/949032)

# 五、总结与展望

该项目的核心，一是通过目标检测获取图像内与中秋节相关的元素，二是根据第一步获取的关键词写古诗。

我在这里给各位准备了几张供大家写诗的图片：

![https://ai-studio-static-online.cdn.bcebos.com/37e7f354b4c54c35953cf8fd5fb800e7f6992a4ff16a40f4a9978c3091dc9a6c](https://ai-studio-static-online.cdn.bcebos.com/37e7f354b4c54c35953cf8fd5fb800e7f6992a4ff16a40f4a9978c3091dc9a6c)
![https://ai-studio-static-online.cdn.bcebos.com/aab1f521b177485d80cb433cb761b5abab10d2514fce4e03b41c2d080574a4d8](https://ai-studio-static-online.cdn.bcebos.com/aab1f521b177485d80cb433cb761b5abab10d2514fce4e03b41c2d080574a4d8)
![https://ai-studio-static-online.cdn.bcebos.com/158f40c25ea548b29e0634095ebc1a1ff4a5d1850682447d9a9a921dc6e696ed](https://ai-studio-static-online.cdn.bcebos.com/158f40c25ea548b29e0634095ebc1a1ff4a5d1850682447d9a9a921dc6e696ed)
![https://ai-studio-static-online.cdn.bcebos.com/520c33eab8cd40b1865bbcc4da8af0ef5f3062d5b11d4a0989eee575cb2f989c](https://ai-studio-static-online.cdn.bcebos.com/520c33eab8cd40b1865bbcc4da8af0ef5f3062d5b11d4a0989eee575cb2f989c)

最后也欢迎大家更换自己的图片，写一首古诗吧！


# 六、个人介绍
- 北京联合大学 机器人学院 自动化专业 2018级 本科生 郑博培
- 百度飞桨开发者技术专家 PPDE
- 深圳柴火创客空间 认证会员
- 百度大脑 智能对话训练师

来AI Studio互粉吧 等你哦  https://aistudio.baidu.com/aistudio/personalcenter/thirdview/147378

欢迎大家fork喜欢评论三连，感兴趣的朋友也可互相关注一下啊~

