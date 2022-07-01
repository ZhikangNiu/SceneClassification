# Scene Classification
一、文件含义：
1. filelist: 标注文件夹
   1. txt_filelist: Place365原始数据标注文件
   2. json_filelist: 将.txt格式的Place365数据标注转换为Json格式，
                    具体使用：使用txt2json.py和val_txt2json.py进行转换
2. train_checkpoint: 权重默认保存路径
3. dataset: 包含txtDataset和jsonDataset两种数据标注下的数据读取
4. level2_acc: 测量二级标签的精确度及其指标
5. train 和 train_timm: 将会在后期合并，是训练文件
6. utils 文件移动操作
7. predict 和 predict_level2: 预测接口
8. config 和 config_new: 超参数配置

二、使用方法
1. train
参数包括以下参数，其中CLASS_NUM参数为必需参数
```python
    parser.add_argument('--batch_size',type=int,default=32,help="input batch size,default = 64")
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs to train for, default=10')
    parser.add_argument("--seed",type = int,default=66,help="random seed")
    parser.add_argument("--class_num",type=int,required=True,help="classification category")
    parser.add_argument("--tensorboard_path",type=str,default="./runs/log")
    parser.add_argument("--img_size",type=tuple,default=(224,224),help="input image size")
    parser.add_argument("--show",action="store_true",default=False)
    parser.add_argument("--checkpoint_dir",type=str,default="./checkpoint")
    parser.add_argument("--log_file",type=str,default="./train_logger.txt")
    parser.add_argument("--GPUS",type=int,default=1)
```
```bash
python train_timm.py --batch_size 32 --epochs 30 --class_num CLASS_NUM
```
1. evaluate
2. predict


三、后期计划