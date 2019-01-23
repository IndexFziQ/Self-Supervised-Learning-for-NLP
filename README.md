# Research on Self-Supervised Learning

A research on self-supervised learning with the interest of applying it into NLP field. Inspired by the talk (Naiyin Wang), this work lists some typical papers about self-supervised learning. The aim is to find the advantage of it and apply it into NLP research. Also, lots of related works will be linked, such as Lecun's presentation and awesome-self-supervised-learning.

## Introduction

First, the definition of self-supervised learning should be given. Three questions are needed to be answered (ref. Wang):

* What is self-supervised learning?
	* Use naturally existed supervision signals for training.
	* (Almost) no human intervention
* Why do we need it?
	* Representation learning is fundamental and necessary in deep learning. We usually use pretrain-then-finetune pipeline.
	* Self-supervised learning can leverage self-labels for representation learning.
* How can we realize it?
	* Context
	* Video
	* Cross-Modality
	* Examplar Learning
	
The following is the details for the 3th question. Also, the difference between self-supervised learning and supervised learning, unsupervised learning, semi-supervised learning, self-taught learning... should gain attention. The definition must be distinguished.

## What is self-supervised learning?

**Compared to other machine learning paradigms:**

**Polarization**

1. supervised learning: Given data and desired output, the model aims to finish the task.
2. unsupervised learning: No guidance at all. Only data with no label. We need to mine useful or meaningful information from the data distribution.

**In Between**

3. semi-supervised learning: Mix labeled and unlabeled data, and usually labeded data is low-source. We utilize large-scale unlabeled data distribution to help supervised learning.
4. self-taught learning: Compared to semi-supervised learning, the unlabeled data distribution can be different from the labeled data. Usually, we use autoencoders to train unsupervised inner features from unlabeled data. Then, use the feature to replace or enhance the input representation of labeled data. Reconstruction can hardly represent semantic information due to using no structural loss.
5. weakly-supervised learning: It is a special supervised learning which use somewhat coarse or inaccurate supervision. Design some simple form label for data. E.g., given scribble, infer the full pixel level segmentation.

**After giving a nearly whole distinction, we can answer "What is self-supervised learning?" clearly.**

6. self-supervised learning: It belongs to unsupervised learning with (almost) no human intervention. Differently, the model uses naturally existed supervision signals for training. Labels are obtained from the structure of the data. In other words, the supervision signals exist in the unlabeled data and we utilize the prediction way (like supervised learning) to train the model.

**Others**

7. transfer learning: Train on one problem, but test on a different but related problem, which is a pretrain-then-finetune pipeline, e.g. multi-task learning, incremental learning and domain adaptation. And to some degree, self-supervised learning is also included because of the pretrain-then-finetune pipeline.
8. active learning: In the beginning, given little labels, the model should study which labeled samples are needed and meaningful. It aims to teach model to select more important data.
9. zero/one/few-shot learning: Given no/one/little samples, we try the best to train a classifier.

At last, reinforcement learning is different from the above. Based on the reward from environment, the model gives the reaction. Compared to self-supervised learning, the feedback is low. 

## Why do we need it?

No matter what CV or NLP with deep learning, representation learning is a fundamental problem. We usually use pretrain-then-finetune pipeline. As to this point, self-supervised learning can leverage self-labels for representation learning. Also, the representation contains high-level semantic information because of the human-like learning mode, which is meaningful for deep learning.

## How can we realize it?

Keep updating ...

### Context

#### Solving the Jigsaw

* **Unsupervised Visual Representation Learning by Context Prediction.** *Carl Doersch, Abhinav Gupta, and Alexei A. Efros.* In ICCV 2015. [[pdf]](https://arxiv.org/pdf/1505.05192.pdf)
    * Motivation
        * Predict relative positions of patches.
    * Contribution
        * The first work to implement self-supervised learning. With the intention to solve the Jigsaw, we need to understand the object firstly.
    * Overview
        * CNN is especially good at it. The paper divided one object to nine patches to predict relative positions.
<div align=center>
    <img src="./images/Jigsaw 1.png" height="50%" width="50%" />
</div>
    
* **Unsupervised learning of visual representations by solving jigsaw puzzles.** *Noroozi, Mehdi and Favaro, Paolo.* In ECCV 2016. [[pdf]](http://arxiv.org/abs/1603.09246)
    * Motivation
        * Use stronger supervision, solve the real jigsaw problem. 
    * Contribution
        * Introduce the context-free network (CFN), a CNN whose features can be easily transferred between detection/classification and Jigsaw puzzle reassembly tasks.
    * Overview
	<div align=center>
	    <img src="./images/Jigsaw 2.png" height="50%" width="50%" />
	</div>
    
#### Colorization

* **Context Encoders: Feature Learning by Impainting.** *Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A.* In CVPR 2016. [[pdf]](https://people.eecs.berkeley.edu/~pathak/papers/cvpr16.pdf)
    * Motivation
        * Given an image with a missing region, we train a convolutional neural network to regress to the missing pixel values. It is possible to learn and predict this structure.
	<div align=center>
	    <img src="./images/Color 1-0.png" height="50%" width="50%" />
	</div>
	
    * Contribution
        * The model consists of an encoder capturing the context of an image into a compact latent feature representation and a decoder which uses that representation to produce the miss- ing image content.
        * Introduce a channel- wise fully-connected layer, which allows each unit in the decoder to reason about the entire image content.
        * With the advancement of adversarial loss.
	
    * Overview
        * The overall architecture is a simple encoder-decoder pipeline. The encoder takes an input image with missing regions and produces a latent feature representation of that image. The decoder takes this feature representation and produces the missing image content.
	<div align=center>
	    <img src="./images/Color 1-1.png" height="50%" width="50%" />
	</div>

* **Colorful Image Colorization.** *Zhang, Richard and Isola, Phillip and Efros, Alexei A.* In ECCV 2016. [[pdf]](https://arxiv.org/abs/1603.08511)

    * Motivation
        * Given a grayscale photograph as input, this paper attacks the problem of hallucinating a plausible color version of the photograph. You have to know what the object is before you predict its color. E.g. Apple is red/green, sky is blue, etc.
	<div align=center>
	    <img src="./images/Color 2-0.png" height="50%" width="50%" />
	</div>
	
    * Contribution
        * propose a fully automatic approach that produces vibrant and realistic colorizations.
        * The method successfully fools humans on 32% of the trials, significantly higher than previous methods. 
        * It shows that colorization can be a powerful pretext task for self-supervised feature learning, acting as a cross-channel encoder. 
	
    * Overview
	<div align=center>
	    <img src="./images/Color 2-1.png" height="50%" width="50%" />
	</div>

