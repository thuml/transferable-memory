# TMU (ICML 2020)
A PyTorch implementation of TMU [[paper](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-memory-icml20.pdf)], a differentiable
framework named transferable memory, which adaptively distills knowledge from a bank of memory states of multiple pretrained RNNs, and applies it
to the target network via a novel recurrent structure called the Transferable Memory Unit (TMU).

Video prediction networks have been used for precipitation nowcasting, early activity recognition, physical scene understanding, model-based visual planning, and unsupervised representation learning of video data.

## Get Started
1. Install Python 3.7, PyTorch 1.3, and OpenCV 3.4.  

2. Download data. This repo contains code for two datasets: the [Moving Mnist dataset](https://1drv.ms/f/s!AuK5cwCfU3__fGzXjcOlzTQw158) and the [KTH action dataset](http://www.nada.kth.se/cvap/actions/).  

3. Train the model. You can use the following bash script to train the model. The learned model will be saved in the `--save_dir` folder. 
The generated future frames will be saved in the `--gen_frm_dir` folder.  
```
cd script/
sh tmu.sh
```



## Related Publication
**PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning.**  
Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang, and Philip S. Yu.  
ICML 2018 [[paper](http://proceedings.mlr.press/v80/wang18b.html)] [[code](https://github.com/Yunbo426/predrnn-pp)]

## Contact
You may send email to yaozy19@mails.tsinghua.edu.cn, yunbo.thu@gmail.com or longmingsheng@gmail.com. 

 
