

8.12
测试了搜集一个S3的real robot版本然后training，效果一般。主要的结论是
1. 实际data很可能不展示出我所assume的那种贴边的特性，比如避障，我simulation里想的是会贴着障碍物走，但human demo完全可以里的很远并且随意改变距离。这种贴边性被削弱了。despite that，如果让搜集demo的时候刻意多搜集一些这种比较贴边的轨迹（因为比较informative），发现还是能得到不等式约束的。这个事情可以被解释为demon的问题
2. 理论上讲不同的feature应该使用不同的阈值，比如line dist这个feature发现抖动很小，很容易学到等式约束。  相反velocity抖动很大，用同样的阈值，很难得出结论其被等式约束。这个事情与归一化无关，归一化主要考虑的是跨stage的feature变化，但是没有衡量单个stage内feature的噪音。但是这样会搞得比较麻烦。是否存在更好的方法.:  没有，缺乏其他信息的情况下，无法判断某个feature的波动来自于噪音还是来自于unconstrained。
3. 某些feature约束很难demonstration，尤其是速度相关的，demonstrator很难让速度稳定，更难呈现出simulation时我用的那种有规律的曲线。所以要么这部分使用oracle controller，要不就干脆不让任务里出现这种约束。也就是简单任务+human dmeo，在做一个复杂任务+oracle demo。

关于算法改进需要研究的地方
1. 目前在短分段，更窄的等式约束，和长分段，更宽的等式约束中更容易选择前者，因为段长*score比较高。这会导致容易学到很短的分段。我之前搞了个 短段惩罚，但是感觉不太好用。这一点需要进一步研究，到底怎么计算loss能够考虑两种效应
2. progress在所有数据集中几乎都是完全与cutpoints无关的，也就是说完全没发挥作用。其功能需要优化
3. 老问题，到底用什么来model 不等式约束。以及等式和不等式约束如何平衡。搞得权重太多如何平衡

关于constraint 的identifiability，需要重新claim一下一下
在 problem formulation 里加入一个明确的 constraint-identifiability / expressiveness assumption；
把 “recover the true constraints from feasible demonstrations” 稍微改成 “recover the latent stage-wise constraint model within the prescribed hypothesis class”；
在 limitation 里明确承认 feasible-only demonstrations 无法一般地区分 true task requirements 与 behaviorally induced regularities；
把 cross-demonstration consistency 和 transfer planning 强调成支持 constraint interpretation 的 evidence，而不是 constraint existence 的证明；
最好针对你已经观察到的 S4 false-positive speed upper bound 直接解释成这个 limitation，而不要仅仅当成一个随机错误。



8/13
1.把思考的em reformulation整理出来了，考虑换成这个架构，似乎更漂亮

固定分段然后模型基本能给出合理的解释，但是依赖超参，并且目前的demo质量不行，只有一半的demo能够在一阶段推断出obs 约束

自由分段，目前会出现一阶段容易错误的吧obs 不当做约束，导致concensus变成0，然后也无法扭转。陷入local optimal了。这个问题也可以算成demonstration的质量问题。就目前这些数据，我感觉也就这样了。


2. 已经测试了reform的新板，采用了共享的constraint 做block 优化。不在给每个demon分配单独的param（这样会导致模型过于复杂）。
好处是现在可以不需要指定不等式和等式，从四种mode里面选择。

目前的问题是    
1. inactive 的分布需要研究一下怎么设置。我目前用的高斯，有的时候偏弱
2. 目前不等式用的soft halt t偏强，它可以把某一边做成单边support，就能节约很多support。
3. 现在不等式约束容易被错误识别，这也没办法
