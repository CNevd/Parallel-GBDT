# Parallel-GBDT
[Parallel Gradient Boosting Decision Trees](http://zhanpengfang.github.io/418home.html)

对于随机森林来讲，可以并行的创建每一棵树，但由于gbdt每棵树之间有依赖关系（梯度下降方向）因此选择在建每棵树内部进行并行
顺便啰嗦一句，RF的bagging其实分两部分，一是instance的bagging，即每一颗树采用的样本是随机bagging产生的，二是对于每一个node
（每一棵树也无所谓喽）随机抽取m个特征然后从中选取分割点（假设特征总数共M且M>>m）

Sequential Decision Tree Building
对于顺序创建gbdt来讲每棵树以及每个节点都是依次处理的
假设输入数据为  N个instances 每个instances有m个特征 + 一个目标（下降目标）

算法需要遍历这一层级的每一个node，对于每个node执行一个查找分割点的过程，该过程需要遍历每个feature来找到最佳分割点。复杂度O(N*m)且不计样本排序及线性搜索

如何实现并行？

方式1：每一层的node间并行（Parallelize Node Building at Each Level）
显而易见在建树过程中每一层级中的node都是相互独立的可以直接并行，但是会引起数据加载不均衡的问题workload imbalanced
（决策树需要剪枝，导致一些节点是要被减掉的，如此一来会出现一些节点下的样本很多，另一些很少），比如下图红框中的四个节点第1个和第3个节点的instances数明显少于第2和第4
据说 该方式效率提升不明显

![image](https://github.com/CNevd/Parallel-GBDT/blob/master/pic/node.png)

方式2：每个node上查找split分割点时候并行（node内部处理并行，Parallelize Split Finding on Each Node）
也很明显，无非两个for循环的并行，方式1第一个for循环，方式2第二个for循环
对于每个feature分别并行执行【将instance按照特征value排序，然后线性搜索，大家都是一个node因此instances都是一样的】，但这样也存在问题：对一些小的节点会存在过多的开销---怎么理解呢？如果一颗树建的越来越深那么分到一个node上的instances就会越来越少，我们称之为小节点，这样再对特征split过程进行并行的话。并行带来的收益可能还cover不了数据切换以及进程通信等带来的开销，如此一来随着建树越来越深并行化就体现不出其优势了。however，这是一个好的方向

![image](https://github.com/CNevd/Parallel-GBDT/blob/master/pic/2.png)

方式3：在每一层对特征进行并行化(Parallelize Split Finding at Each Level by Features)
你会发现，哎呦，每一层对特征并行化，node去哪了？？？？莫急。。。
在顺序建树的过程中我们发现，整个过程是通过两个循环得到的，第一个循环遍历所有叶节点，第二个遍历在每个叶节点内部遍历所有特征
方式3采用颠倒两个for循环的策略，并行地去找同一层的特征分割点，只在建树开始的时候对所有特征的所有value进行排序，在遍历过程中记录对于某一特征fid其各个样本所对应的node（代码中的pos）

![image](https://github.com/CNevd/Parallel-GBDT/blob/master/pic/3.png) ![image](https://github.com/CNevd/Parallel-GBDT/blob/master/pic/3_matrix.png)

