# Equalization Loss for Long-Tailed Object Recognition
Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, 
Wanli Ouyang, Changqing Yin, Junjie Yan

[[`arXiv`](https://arxiv.org/abs/)] [[`BibTeX`](#CitingEQL)]

<div align="center">
  <img width="70%", src="https://tztztztztz.github.io/images/eql-gradient.jpg"/>
</div><br/>

In this repository, we release code for Equalization Loss (EQL) in Detectron2. EQL protects the learning for rare categories from being at a disadvantage during the network parameter updating under the long-tailed situation.

## Installation
Install Detectron 2 following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). You are ready to go!

## LVIS Dataset

Following the instruction of [README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) to set up the lvis dataset.


## Training

To train a model with 8 GPUs run:
```bash
cd /path/to/detectron2/projects/EQL
python train_net.py --config-file configs/eql_mask_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/EQL
python train_net.py --config-file configs/eql_mask_rcnn_R_50_FPN_1x.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

# Pretrained Models
 
## Instance Segmentation on LVIS

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom", align="left">Model</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP.r</th>
<th valign="bottom">AP.c</th>
<th valign="bottom">AP.f</th>
<th valign="bottom">AP.bbox</th>
<th valign="bottom">model</th>

<!-- TABLE BODY -->
<tr><td align="left">R50-FPN-Mask</td>
<td align="center">21.2</td>
<td align="center">3.2</td>
<td align="center">21.1</td>
<td align="center">28.7</td>
<td align="center">20.8</td>
<td align="center"><a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a></td>
</tr>
<tr><td align="left">R50-FPN-Mask-EQL</td>
<td align="center">24.0</td>
<td align="center">9.4</td>
<td align="center">25.2</td>
<td align="center">28.4</td>
<td align="center">23.6</td>
<td align="center"><a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a></td>
</tr>
<tr><td align="left">R50-FPN-Mask-EQL-Resampling</td>
<td align="center">26.1</td>
<td align="center">17.2</td>
<td align="center">27.3</td>
<td align="center">28.2</td>
<td align="center">25.4</td>
<td align="center"><a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a></td>
</tr>

</tbody></table>

Note that the final results of these configs have large variance across different runs.


## <a name="CitingEQL"></a>Citing EQL

If you use EQL, please use the following BibTeX entry.

```BibTeX
@InProceedings{tan2020eql,
  title={Equalization Loss for Long-Tailed Object Recognition},
  author={Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, 
  Wanli Ouyang, Changqing Yin, Junjie Yan},
  journal={ArXiv:},
  year={2020}
}
```
