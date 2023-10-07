# Pytorch_Study

### 学习网站：https://www.bilibili.com/video/BV1Y7411d7Ys?p=1&vd_source=91c56b2745a5b892c273f637afde8f44

### 报错问题
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. 报错的解决方法
#### 方法一：在文件首部添加如下代码
```commandline
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
```
#### 方法二：删除所在虚拟环境下的libiomp5md.dll文件
> 如果是在Anaconda的base环境下，删除…\anaconda3\Library\bin\libiomp5md.dll
> 
>如果是在某个env(例如名为work)下：删除…\Anaconda3\envs\work\Library\bin\libiomp5md.dll

