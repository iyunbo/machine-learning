# machine-learning
Content for Udacity's Machine Learning curriculum

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.

SVM

1. 例子： https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-10-16
基于病例的数据，通过建立SVM模型来将病例分类并识别出哪些是糖尿病患者，哪些不是。

2. 
- 由于SVM的特性，分类结果保证是全局优化的而不是局部优化的。
- 对线性可分离和线性不可分离的数据都适用
- 引入核技巧可以有效解决很多非线性的问题

训练数据不多，非线性的情况下可以有比较好的表现

3. 
- 结果不容易被解释和评估
- 训练集大太的情况下，运算时间太长
- 如果特征数量太大，算法的性能可能很低

维度过大或者训练数据量过大不宜使用svm

在预测学生表现的应用中可以考虑使用SVM模型，原因是训练数据不多（几百行）而且数据之间的关系目前难以确定是不是线性的，SVM有比较好的通用型，给出的结果也是全局优化的，选用这个模型似乎可以给出比较合理的答案。
