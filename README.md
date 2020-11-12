# AUSH

code for paper AUSH: Generative Adversarial Network for Augmented Shilling Attack against Recommender Systems


### ijcai实验数据备份

https://pan.baidu.com/s/1q2xCWBWH8RiYHYiGTIULOg
提取码：9fcz

 ---

### step1:数据处理

- amazon的5cores数据集的写好在\AUSH\test_main\data_preprocess.py下的data_preprocess函数，处理其他数据集参考那份代码和下面的说明

 - 数据格式转换为三列[userid,itemid,rating];userid,itemid的起始值为0;rating的类型为float,评分保证在1.0-5.0;
 - 按照9:1划分训练/测试集合;文件名存为/data/data/" + data_set_name + "_train/test.dat";全局随机划分，不用按照用户;我一般用sklearn.model_selection.train_test_split;
 - 数据统计:utils\load_data\load_data.py里有一个读数据的类load_data()，把数据路径放进path_train, path_test,
 （1）记录训练数据集中的用户数self.n_users、物品数self.n_items、评分数，这些最后和实验结果一起汇报给老师
 （2）get_item_pop()函数可以得到所有物品得到的评分数量，接着根据物品评分数量对物品分组


### step2: 目标设置
- 写好在\AUSH\test_main\data_preprocess.py下的exp_select函数，下面是具体说明
 - 选择攻击目标:可以把商品按评分数量分组，每组选几个个，目的是看攻击效果在流行度（评分数量）上是否有区分
 - attack num:在所有数据集上固定为50
 - filler size:训练集中用户的平均评分数
 - selected items和target users
 - 全局热门top作为bandwagon attack的设置
 ```
 random attack:无
 average attack:无
 segement attack:为每个攻击目标选择3个selected items和50个target users。selected items是目标商品同类别的热门商品，数据集没有类别信息，就在全局热门里随机。target users定义为对selected items评高分对目标商品无评分的用户。参考["../data/data/filmTrust_selected_items", "../data/data/filmTrust_target_users"]的格式存储
 bandwagon attack:计算全局热门top3作为所有攻击目标的selected item。

 ```


### step3. 训练和评估

 - step1:生成baseline攻击模型的攻击文件
 ```shell script
python main_baseline_attack.py --dataset filmTrust --attack_methods average,segment,random,bandwagon --targets 601,623,619,64,558 --filler_num 36 --bandwagon_selected 103,98,115 --sample_filler 1
```
 - step2:计算baseline攻击的结果(依赖step1)
 ```shell script
//在shell脚本里加model_name和targetid的for循环,nohup并行跑。注意根据服务器资源选择GPU&控制并行数量。
python main_train_rec.py --dataset filmTrust --attack_method segment --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - step3:计算没有攻击时的结果
 ```shell script
//在shell脚本里加model_name和targetid的for循环,nohup并行跑。注意根据服务器资源选择GPU&控制并行数量。
python main_train_rec.py --dataset filmTrust --attack_method no --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - step4:训练gan并生成攻击文件(依赖setp1)
 ```shell script
//在shell脚本里加targetid的for循环,nohup并行跑。注意根据服务器资源选择GPU&控制并行数量。
python main_gan_attack.py --dataset filmTrust --target_ids 601,623,619,64,558 --filler_num 36
````

 - step5:计算gan攻击的结果(依赖step4)
 ```shell script
//在shell脚本里加targetid的for循环,nohup并行跑。注意根据服务器资源选择GPU&控制并行数量。
python main_train_rec.py --dataset filmTrust --attack_method gan --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - step6:攻击方法对比(依赖step2,3,5)
 ```shell script
//攻击效果对比
python main_eval_attack.py --dataset filmTrust --filler_num 36 --attack_methods gan,segment,average --rec_model_names NMF_25 --target_ids 601,623,619,64,558
//真实性对比
python main_eval_similarity.py --dataset filmTrust --filler_num 36 --targets 601,623 --bandwagon_selected 103,98,115
```
 
 <!--
 - （1）计算没有攻击时的推荐结果

 ```
 main_train_rec.py --dataset filmTrust --attack_num 50 --filler_num 36 --attack_method no --model_name NMF_25 --target_id 5

 //数据集确定之后，前三个参数dataset,attack_num,filler_num都固定了,要**改后两个参数model_name和target_id**,遍历NNMF,IAutoRec,UAutoRec,NMF_25个推荐模型和所有的target_id

 //main_train_rec里有is_train，同一个数据集同一个推荐模型，先is_train训练一个目标，下面的target的is_train=0.（1）is_train=1，用第一个目标item并行训练所有的NNMF,IAutoRec,UAutoRec,NMF_25（2）is_train=0，用训练好的四个模型得到剩下的结果
 ```

 - （2）生成baseline攻击模型的攻击文件

```
main_baseline_attack.py --dataset grocery --targets 所有targetid用逗号隔开 --attack_num 50 --filler_num 36 --write_to_file 1

//dataset_class.get_all_mean_std()会计算所有item的平均分，数据集大的话很慢，如果要重复使用可以优化一下
```

 - （3）生成gan攻击模型的攻击文件
```
main_gan_attack.py --dataset grocery --target_id 5 --attack_num 50 --filler_num 36 --filler_method 0 --write_to_file 1

//并行跑所有target id
```

 - （4）计算攻击后的推荐结果
```
main_train_rec.py --dataset grocery --model_name NMF_25 --attack_method gan --target_id 5 --attack_num 50 --filler_num 36

//并行跑所有model_name/attack_method/target id
```

 - （5）评估攻击效果
```
main_eval_attack.py --dataset grocery --rec_model_name IAUtoRec --attack_method gan --target_id 5 --attack_num 50 --filler_num 36

//并行跑所有model_name/attack_method/target id
```



### 3.3.6 评估攻击效果样例流程
```
//
//计算没有攻击时的推荐结果
main_train_rec.py --dataset filmTrust --model_name NMF_25 --attack_method no --target_id 5 --attack_num 50 --filler_num 36
//生成baseline攻击模型的攻击文件
main_baseline_attack.py --dataset filmTrust --attack_methods segment --targets 5 --attack_num 50 --filler_num 36 --write_to_file 1
//生成gan攻击模型的攻击文件
main_gan_attack.py --dataset filmTrust --target_id 5 --attack_num 50 --filler_num 36 --filler_method 0 --write_to_file 1
//计算攻击后的推荐结果
main_train_rec.py --dataset filmTrust --model_name NMF_25 --attack_method gan --target_id 5 --attack_num 50 --filler_num 36
//评估攻击效果
main_eval_attack.py --dataset filmTrust --rec_model_name IAUtoRec --attack_method gan --target_id 5 --attack_num 50 --filler_num 36
```


----

## 实验二：已有数据集跑新baseline

```
//step1 - 生成baseline攻击模型的攻击文件：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id，loss遍历[0和1]
python main_gan_attack_baseline.py --dataset filmTrust --target_id 5 --attack_num 50 --filler_num 36 --loss 0
//step2 - 计算攻击后的推荐结果：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id，attack_method[G0和G1],model_name遍历所有推荐模型
main_train_rec.py --dataset filmTrust --model_name NMF_25 --attack_method G0 --target_id 5 --attack_num 50 --filler_num 36
//step3 - 计算没有攻击时的推荐结果：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id,model_name遍历所有推荐模型
main_train_rec.py --dataset filmTrust --model_name NMF_25 --attack_method no --target_id 5 --attack_num 50 --filler_num 36
//step4 - 评估攻击效果：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id，attack_method[G0和G1],model_name遍历所有推荐模型
main_eval_attack.py --dataset filmTrust --rec_model_name IAUtoRec --attack_method gan --target_id 5 --attack_num 50 --filler_num 36
//step4的结果保存
```
---


##
### 1.环境配置
 - 我的环境导出在conda_env_chensi.yaml,在cmd下conda env create -f conda_env_chensi.yaml导入环境
 - ps:这是我目前的环境，可以跑这份代码。但其中比如suprise（推荐系统库，NMF的训练需要）库好像没有导出(不知道为啥)需要再安装,方法是pip install scikit-surprise，如果提示缺少MS C++，可以参考https://stackoverflow.com/questions/29846087/microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat/51087608#51087608 的解决办法，下载 http://go.microsoft.com/fwlink/?LinkId=691126&fixForIE=.exe. 再pip install就可以解决。
### 2.代码结构
可以看到这份代码分为5个文件夹
 - data:存放数据，其中data/data里是原始数据，data/data_attack是攻击后的数据
 - result:result/model_ckpt中存放训练好的模型的ckpt，可复现论文里的结果，pred_result里存放攻击后target item的hit ratio和预估分
 - utils:放一些读写数据或日志的脚本
 - model:存放模型的python脚本，model/attack_model里是baseline攻击模型和gan attack也就是我们的GAN攻击，autorec和nnmf是推荐模型
 - test_main:存放训练模型的python脚本，包括训练gan,生成攻击文件,攻击,评估攻击效果,**是连芸主要用到的**。
    1. main_attack_baseline.py:用baseline模型生成攻击文件
    2. main_gan_attack.py:训练gan并生成攻击文件
    3.main_train_rec.py:训练推荐模型并生成推荐结果
    4.main_eval_attack.py:评估攻击效果 - 在目标用户和全局用户上的shift和HitRatio
    5.main_eval_similarity.py:评估攻击的不易检测性 - 对比攻击前后item评分向量的分布的距离，以及JS和TVD距离，这份代码还不能改一两个参数就能直接跑，我改一哈。
-->



----

## TIPS
 - 服务器上并行跑python的脚本样例更新在example.sh
 
 
 ----






```
//step1 - 生成baseline攻击模型的攻击文件：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id，loss遍历[0和1]
python main_gan_attack_baseline.py --dataset filmTrust --target_id 5 --attack_num 50 --filler_num 36 --loss 0
//step2 - 计算攻击后的推荐结果：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id，attack_method[G0和G1],model_name遍历所有推荐模型
main_train_rec.py --dataset filmTrust --model_name NMF_25 --attack_method G0 --target_id 5 --attack_num 50 --filler_num 36
//step3 - 计算没有攻击时的推荐结果：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id,model_name遍历所有推荐模型
main_train_rec.py --dataset filmTrust --model_name NMF_25 --attack_method no --target_id 5 --attack_num 50 --filler_num 36
//step4 - 评估攻击效果：dataset遍历[filmTrust和ml100k]，filler_num跟着改，target_id遍历每个数据集对应的target id，attack_method[G0和G1],model_name遍历所有推荐模型
main_eval_attack.py --dataset filmTrust --rec_model_name IAUtoRec --attack_method gan --target_id 5 --attack_num 50 --filler_num 36
//step4的结果保存
```





