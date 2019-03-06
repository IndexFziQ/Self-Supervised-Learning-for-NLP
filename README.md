# Research on Self-Supervised Learning

A research on self-supervised learning with the interest of applying it into NLP field. Inspired by the talk (Naiyan Wang), this work lists some typical papers about self-supervised learning. The aim is to find the advantage of it and apply it into NLP research. Also, lots of related works will be linked, such as Lecun's presentation and awesome-self-supervised-learning.

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

6. self-supervised learning: Like unsupervised learning, (almost) no human intervention. Differently, the model uses naturally existed supervision signals for training. Labels are obtained from the structure of the data. In other words, the supervision signals exist in the unlabeled data and we utilize the prediction way (like supervised learning) to train the model.

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
	    <img src="./Images/Jigsaw 1.png" height="50%" width="50%" />
	</div>
    
* **Unsupervised learning of visual representations by solving jigsaw puzzles.** *Noroozi, Mehdi and Favaro, Paolo.* In ECCV 2016. [[pdf]](http://arxiv.org/abs/1603.09246)
    * Motivation
        * Use stronger supervision, solve the real jigsaw problem. 
    * Contribution
        * Introduce the context-free network (CFN), a CNN whose features can be easily transferred between detection/classification and Jigsaw puzzle reassembly tasks.
    * Overview
	<div align=center>
	    <img src="./Images/Jigsaw 2.png" height="50%" width="50%" />
	</div>
    
#### Colorization

* **Context Encoders: Feature Learning by Impainting.** *Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A.* In CVPR 2016. [[pdf]](https://people.eecs.berkeley.edu/~pathak/papers/cvpr16.pdf)
    * Motivation
        * Given an image with a missing region, we train a convolutional neural network to regress to the missing pixel values. It is possible to learn and predict this structure.
	<div align=center>
	    <img src="./Images/Color 1-0.png" height="50%" width="50%" />
	</div>
	
    * Contribution
        * The model consists of an encoder capturing the context of an image into a compact latent feature representation and a decoder which uses that representation to produce the miss- ing image content.
        * Introduce a channel- wise fully-connected layer, which allows each unit in the decoder to reason about the entire image content.
        * With the advancement of adversarial loss.
	
    * Overview
        * The overall architecture is a simple encoder-decoder pipeline. The encoder takes an input image with missing regions and produces a latent feature representation of that image. The decoder takes this feature representation and produces the missing image content.
	<div align=center>
	    <img src="./Images/Color 1-1.png" height="50%" width="50%" />
	</div>

* **Colorful Image Colorization.** *Zhang, Richard and Isola, Phillip and Efros, Alexei A.* In ECCV 2016. [[pdf]](https://arxiv.org/abs/1603.08511)

    * Motivation
        * Given a grayscale photograph as input, this paper attacks the problem of hallucinating a plausible color version of the photograph. You have to know what the object is before you predict its color. E.g. Apple is red/green, sky is blue, etc.
	<div align=center>
	    <img src="./Images/Color 2-0.png" height="50%" width="50%" />
	</div>
	
    * Contribution
        * propose a fully automatic approach that produces vibrant and realistic colorizations.
        * The method successfully fools humans on 32% of the trials, significantly higher than previous methods. 
        * It shows that colorization can be a powerful pretext task for self-supervised feature learning, acting as a cross-channel encoder. 
	
    * Overview
	<div align=center>
	    <img src="./Images/Color 2-1.png" height="50%" width="50%" />
	</div>

### Video

#### Motion consistency

* **Unsupervised learning of visual representations using videos**. *Wang, Xiaolong and Gupta, Abhinav.*  In ICCV 2015. [[pdf]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Unsupervised_Learning_of_ICCV_2015_paper.pdf)
    * Motivation
        * Find corresponding pairs using visual tracking.
    * Contribution
        * Define a rank task to find corresponding two frames.
    * Overview

<div align=center>
	    <img src="./Images/video1.png" height="50%" width="50%" />
</div> 

* **Dense optical flow prediction from a static image**. *Jacob Walker, Abhinav Gupta, and Martial Hebert*. In ICCV 2015. [[pdf]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Walker_Dense_Optical_Flow_ICCV_2015_paper.pdf)
    * Motivation
        * Directly predict motion,  Motion is not predictable by its nature.
    * Contribution
        * The ultimate goal is not to predict instance motion, but to learn common motion of visually similar objects.
    * Overview
<div align=center>
	    <img src="./Images/video2.png" height="50%" width="50%" />
</div> 

* **Pose from Action: Unsupervised Learning of Pose Features based on Motion**. *Senthil Purushwalkam and Abhinav Gupta*.  In ECCVW 2016. [[pdf]](https://arxiv.org/pdf/1609.05420.pdf)
    * Motivation
        * Similar pose should have similar motion. 
    * Contribution
        * Learning appearance transformation.
    * Overview

<div align=center>
	    <img src="./Images/video3.png" height="50%" width="50%" />
</div> 

<div align=center>
	    <img src="./Images/video4.png" height="50%" width="50%" />
</div> 

#### Action Order

* **Shuffle and learn: unsupervised learning using temporal order verification**. *Misra, Ishan and Zitnick, C. Lawrence and Hebert, Martial*. In ECCV 2016. [[pdf]](https://arxiv.org/pdf/1603.08561.pdf)
    * Motivation
        * Is the temporal order of a video correct? 
    * Contribution
        * Encode the cause and effect of action.
    * Overview

<div align=center>
	    <img src="./Images/video5.png" height="50%" width="50%" />
</div> 

* **Self-Supervised Video Representation Learning With Odd-One-Out Networks**. *Fernando, Basura and Bilen, Hakan and Gavves, Efstratios and Gould, Stephen*. In CVPR 2017. [[pdf]](https://arxiv.org/pdf/1611.06646.pdf)
    * Motivation
        * Is the temporal order of a video correct? 
    * Contribution
        * Define the task to find the odd sequence.
    * Overview

<div align=center>
	    <img src="./Images/video6.png" height="50%" width="50%" />
</div> 

### Cross-Modality

* **TextTopicNet - Self-Supervised Learning of Visual Features Through Embedding Images on Semantic Text Spaces.** *Patel et al.* In CVPR 2017. [[pdf]](https://arxiv.org/pdf/1807.02110.pdf)
    * Motivation
        * Take advantage of multi-modal context (Wikipedia) for self-supervised learning.
    * Contribution
        * Train a CNN to predict the more probable pic to appear as an illustration.
        * SOTA performance in image classification, object detection, and multi-modal retrieval.
    * Overview

<div align=center>
	    <img src="./Images/cross1.png" height="50%" width="50%" />
</div>


# Reference

1. The Presentation given by Yann LeCun in the Opening of IJCAI 2018: We Need a World Model.[ [pdf](https://cloud.tencent.com/developer/article/1356966), [chinese](http://ir.hit.edu.cn/~zyli/papers/lecun_ijcai18.pdf) ]
2. awesome-self-supervised-learning. [ [url](https://github.com/jason718/awesome-self-supervised-learning) ]
3. A Survey to Self-supervised learning. [ [ppt](http://link.zhihu.com/?target=http%3A//winsty.net/talks/self_supervised.pptx) ]
