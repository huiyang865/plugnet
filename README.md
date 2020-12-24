# PlugNet: Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit

PlugNet combines the pluggable super-resolution unit (PSU) to solve the low-quality text recognition from the feature-leve. The research paper can be found [here](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600154.pdf). The presentation of this paper refer to [TechBeat](https://www.techbeat.net/talk-info?id=450).

![PlugNet Overview](overview.png)

Note: Due to business competition, we only open some relevant core code for reference and communication of relevant researchers. Pretraining models and specific training methods cannot be provided in open source at present.

## Reproduced results

|               | IIIT5k |  SVT |  IC03 |  IC13 |  SVTP |  CUTE |
|:-------------:|:------:|:----:|:-----:|:-----:|:-----:|:-----:|
| ASTER.Pytorch |  93.2  | 89.2 | 92.2  |   91  |  81.2 |  81.9 |
| Aster(our training) |  93.4  | 89.5 | 94.5  |   91.8  |  78.5 |  79.5 |
| PlugNet |  94.4  | 92.3 | 95.7  |   95.0  |  84.3 |  85.0 |


You can use the codes to bootstrap for your next text recognition research project.

## Citation

If you find this project helpful for your research, please cite the following papers:

```
@article{eccv2020plugnet,
  author  = {Yongqiang Mou and
               Lei Tan and
               Hui Yang and
               Jingying Chen and
               Leyuan Liu and
               Rui Yan and
               Yaohong Huang},
  title   = {PlugNet: Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit},
  journal = {The 16th European Conference on Computer Vision (ECCV 2020), 2020.}, 
  volume  = {}, 
  number  = {}, 
  pages   = {1-17},
  year    = {2020}, 
}
```

IMPORTANT NOTICE: Although this software is licensed under MIT, our intention is to make it free for academic research purposes. If you are going to use it in a product, we suggest you [contact us](huiyang865@hotmail.com) regarding possible patent issues.