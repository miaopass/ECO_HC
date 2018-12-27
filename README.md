#ECO_HC
2018/12/27修改：之前的版本CN特征的文件路径是我自己电脑的路径。虽然在linux下可以正常运行，因为会读取一个空的cn矩阵，但是在windows下就会报错。
另外删除了两个没有用上的包。
目前只要安装了cmake和opencv的linux系统电脑都可以运行。
windows下，暂未测试。

分割线
-----------------------------------------------------------------------------
ECO原作者github地址：https://github.com/martin-danelljan/ECO,源代码为matlab

此代码是在https://github.com/nicewsyly/ECO下进行修改。
该作者用c++，opencv复现了ECO的代码，但是其中有部分错误，导致实际跟踪效果较差。

本人修改了其中的错误，主要有eco_sample_update.cpp下的大量错误代码，以及部分小错误。

另外添加了CN特征的提取，以及fDSST用来作尺度变换。复现了ECO的HC版本。
在本人的电脑上(4  Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz)，小目标能达到60+fps，大目标40+fps。
代码中部分参数不可调（因为未复现完）。
为了方便使用去掉了caffe的部分，想要有cnn特征的可以联系email：21710134@zju.edu.cn。

如何使用次代码：

sudo apt-get install cmake

git clone git@github.com:miaopass/ECO_HC.git

cd ECO_HC

cmake .

make

./demo


