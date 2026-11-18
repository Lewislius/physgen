# 设计修订记录

## 本次固定的改变

1. A/B/C/D均从官方基础开始独立run，不继承任何R10A或其他R版本model/LoRA/optimizer。
2. 每个候选1000个optimizer updates，有效batch4；主干video LoRA始终active。
3. B从“选定原生block全参数开放”改为“选定层rank64、其他层rank32的视频LoRA”；全部native base冻结。
4. A/C/D完整时空执行器保留全宽计算，官方base冻结引用、Expert LoRA独立；不是一千步从随机权重重学大视频网络。
5. 原生文本Q/K/V/O、text_embedding、T5/Qwen/VAE始终冻结；A/B的新semantic reader有独立参数，不算原生文本LoRA。
6. 取消S3扩展训练、S4偏好后训练，以及所有偏好/奖励/新教师/时间差分/RGB额外loss。
7. 只用真实视频FM＋原tail＋早期raw-route辅助，后者320步退出。
8. C不再通过前64步关闭semantic module解决梯度；其内层官方base非零、Wr=identity，仅最终writer零初始化。
9. 所有物理内容保持256×3072，所有FFN执行宽度14336；没有3072→512→3072的唯一物理通路。
10. 文件中的参数数字是主要矩阵预算，需实际代码重算；冻结base共享省存储但不省新增执行FLOPs。

## 不变的研发目标

PC、SA、Joint及真实视频质量共同提升；旧强样本继续提高，新联合成功样本增加。目标PC≥3.85/3.95、SA≥3.25、Joint≥14/40。没有使用较弱目标替代用户目标，也没有提供未经训练的达标承诺。
