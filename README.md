# [第二届计图(Jittor)人工智能挑战赛](https://www.educoder.net/competitions/index/Jittor-3)
## 赛题二：可微渲染新视角生成赛题

## 简介
本项目主要基于由 [jittor](https://github.com/Jittor/jittor) 框架开发的 [JNeRF](https://github.com/Jittor/JNeRF) 实现。

## 安装
本项目的运行环境要求
* Ubuntu 18.04
* Python == 3.7
* Jittor >= 1.3.4.13

**步骤1：安装依赖包**

推荐使用 [conda](https://www.anaconda.com/) 创建环境
```sh
conda create -n jnerf python=3.7
conda activate jnerf
python -m pip install -r requirements.txt
```
如果在安装过程中有任何问题，可以参考 [Jittor](https://github.com/Jittor/jittor).

**步骤2：安装 JNeRF**
```sh
cd python
python -m pip install -e .
```
安装完成之后可以在 Python 解释器中运行 `import jnerf` 测试 JNeRF是否安装成功。

## 模型训练
* **数据获取**

本项目训练数据为 Jittor 大赛提供的合成数据 `nerf_synthetic`，通过运行 `bash download_data.sh` 即可获取。在训练时，必须修改配置文件中的数据路径<dataset_dir>，具体操作在 ./configs/*.py/ 中。

* **训练**
```
python run_net.py --config-file ./configs/Easyship.py 
```
训练配置文件针对数据集中的五个不同场景进行了不同设置，可直接选择不同配置文件进行不同模型的训练，可选择的配置文件在 `configs` 文件夹下，分别为 `Car.py, Coffee.py, Easyship.py, Scar.py, Scarf.py`。本项目训练用到的 GPU 是 `Tesla v100 32G`，单卡训练一个模型的时间大概是三个小时，训练结束会在 `./logs` 文件夹下得到每个模型对用的 `test` 渲染结果，在比赛 `A榜` 和 `B榜` 中得到的结果如下：

A榜

|Scenes| Car (val/test) | Coffee (val/test) | Easyship (val/test) | Scar (val/test) | Scarf (val/test) | Total (test) |  
|----|----|----|----|----|----|----|
|PSNR|22.312/21.3024|38.75/34.859|23.214/20.324|44.82/40.28|30.51/30.9644|152.326|

B榜

|Scenes| Car (val/test) | Coffee (val/test) | Easyship (val/test) | Scar (val/test) | Scarf (val/test) | Total (test) |  
|----|----|----|----|----|----|----|
|PSNR|22.312/18.253|38.7526.8729|23.214/17.2914|44.82/28.7102|30.51/28.0804|119.2079|

## 测试

训练结束之后会在 `./logs/<场景名>/` 下自动保存参数文件 `*.pkl`. 也可以加载预训练模型进行测试和渲染，预训练参数模型可从[此链接](https://www.aliyundrive.com/s/gDkZVdpwum5)下载，下载完成后在根目录下新建 `./log` 
文件夹，并将参数文件组织成以下形式

```
--logs
  |--Car
  |  |--params.pkl
  |--Coffee
  |  |--params.pkl
  |--Easyship
  |  |--params.pkl
  |--Scar
  |  |--params.pkl
  |--Scarf
  |  |--params.pkl
```

可以运行以下代码进行测试，只对 `<文件名>` 进行修改即可。
```
python run_net.py --config-file ./configs/<文件名>.py --task test
```

运行结束会在 `./result/<文件名>/test` 文件夹下得到相应场景的测试视角图片。

也可以运行以下代码，运行结束会`./result/<文件名>/` 生成多视角展示的 `.mp4` 格式的视频
```
python run_net.py --config-file ./configs/<文件名>.py --task render
```

## 引用


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}