# SSL-MI203
Semi Supervised Learning project made as final evaluation in MI203 - Machine Learning at ENSTA Paris. Group members are: Filipe LACERDA, Daniel FRULANE, Eduardo GUIMARAES.

The group was to read [[1]](#1), understand and implement a version of the FixMatch algorithm using the CIFAR-10 dataset. One of the tasks also involved comparing the algorithm to ther more known models, like Resnet18 and AlexNet.

## Semi Supervised Learning and the FixMatch Algorithm

SSL(Semi Supervised Learning) is the process of using unlabeled data to train and improve our model. Much of the need of such method comes from the fact fully proper and labeled training data sets might be hard to come by in certain applications. FixMatch itself is a simplification of existing SSL methods.

The algorithms main ideas can be summed up as it follows:

1. We with a set of labeled images $B$ and a batch of unlabeled images $\mu B$, where $\mu$ is a hyperparameter determining the proportion between labeled and unlabeled images;

2. We then train the model initially in the labelled images of $B$, using as the loss of this supervised training:

$$
l_{s} = \frac{1}{B} \sum_{b=1}^{B} H(p_{b},p_{m}((y  | \alpha (x_{b})))
$$ 

3. Each unlabeled image receives a weakly augmentation($\alpha$) and a strong augmentation($A$);

4. We use the model to predict each unlabeled image's ,$b$ , class, generating a probability to each one of the possible classes of the problem. If the largest probability of this distribution is above a certain threshold $\tau$, it is considered a pseudo-label $\hat{q}_{b}$ and will be used in the SSL improving of the model;

$$q_{b} = p_{m} (y | \alpha(u_{b}))$$

```math
\hat{q}_{b} = \text{arg\,max}(q_{b})
```

5. The strong augmentation of the images is then predicted, generating a new probability distribution that includes the pseudo-label of the weakly augmented version of the same image. A new loss function, $l_{u}$, is defined as the average of the cross-entropy between the obtained label and the pseudo-label of the weakly augmented image:


```math
l_{u} = \frac{1}{\mu B} \sum_{b=1}^{B} 1(\max(q_{b}) \geq \tau)H\left(\hat{q}_{b},p_{m}(y | A(u_{b}))\right)
```

6. The total loss minimized by the model is:

$$
loss = l_{s} + \lambda_{u}l_{u}
$$

Where $\lambda_{u}$ is a hyperparameter weighting the importance of the unlabeled images in the training process.




## References

<a id="1">[1]</a>
Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C. A., ... & Li, C. L. (2020). Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33, 596-608. 