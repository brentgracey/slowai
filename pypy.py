import fastbook

fastbook.setup_book()
from fastai.vision.all import *
from fastbook import *
path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path

def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res

mnist = DataBlock((ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

dls = mnist.dataloaders(path)

simple_cnn = sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,32),           #2x2
    conv(32,2, act=False), #1x1
    Flatten(),
)

learn = Learner(dls, simple_cnn, loss_func=F.cross_entropy, metrics=accuracy)

learn.fit_one_cycle(2, 0.01)