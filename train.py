# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory.
# This directory will be recovered automatically after resetting environment.
!ls /home/aistudio/data

# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory.
# All changes under this directory will be kept even after reset.
# Please clean unnecessary files in time to speed up environment loading.
!ls /home/aistudio/work


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required,
# you need to use the persistence path as the following:
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries

# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code,
# so that every time the environment (kernel) starts,
# just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')

!pip install paddlex -i https://mirror.baidu.com/pypi/simple


!ls /home/aistudio/data/data65/train-images-idx3-ubyte
!ls /home/aistudio/data/data65/train-labels-idx1-ubyte

# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])

train_dataset = pdx.datasets.VOCDetection(
    data_dir='xdata',
    file_list='xdata/train_list.txt',
    label_list='xdata/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='xdata',
    file_list='xdata/val_list.txt',
    label_list='xdata/labels.txt',
    transforms=eval_transforms)


num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_interval_epochs=20,
    save_dir='output/yolov3_darknet53',
    use_vdl=True)


import glob
paths = glob.glob(r'xdata/testImage/images/*.jpg')
paths.sort()
print(round(1.23456,3))
for path in paths:
image_name = path[23:42]
#


import paddlex as pdx

model = pdx.load_model('output/yolov3_darknet53/best_model')
for path in paths:
    print(path)
    image_name = path[23:38]
    result = model.predict(path)
    for detail in result:
        print(detail)
        #解析score
        score = detail['score']
        bbox = detail['bbox']
        x = int(bbox[0])
        y = int(bbox[1])
        width = int(bbox[2])
        height = int(bbox[3])
        #置信度>0.01才保存
        if score > 0.01:
            newObj = image_name + " " + str(round(score,3)) + " " + str(x) + " " + str(y) + " " + str(x+width) + " " + str(y+height)+'\n'
            print(newObj)
            with open("result_4.txt", "a+", encoding='utf-8') as f:
                f.write(newObj)
                f.close()
print("end")
    #pdx.det.visualize(image_name, result, threshold=0.1, save_dir='./output/yolov3_darknet53')
import paddlex as pdx

model = pdx.load_model('output/yolov3_darknet53/best_model')
image_name = 'xdata/testImage/images/006497801008779.jpg'
pdx.det.visualize(image_name, result, threshold=0.01, save_dir='./output/yolov3_darknet53')

