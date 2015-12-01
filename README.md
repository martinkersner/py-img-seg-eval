# Image Segmentation Evaluation
Martin Keršner, [m.kersner@gmail.com](mailto:.m.kersner@gmail.com)

Evaluation metrics for image segmentation inspired by paper [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/abs/1411.4038).

### Pixel accuracy
<img src="http://i.imgur.com/dJTYzEu.png?1" />

### Mean accuracy
<img src="http://i.imgur.com/Ldz3wXu.png?1" />

### Mean IU
<img src="http://i.imgur.com/nOvJZXw.png?1" />

### Frequency Weighted IU
<img src="http://i.imgur.com/wx9YnI5.png?1" />

#### Explanatory notes
* *n_cl* : number of classes included in ground truth segmentation
* *n_ij* : number of pixels of class *i* predicted to belong to class *j*
* *t_i*  : total number of pixels of class *i* in ground truth segmentation
