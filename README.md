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

6. self-supervised learning: It belongs to unsupervised learning with (almost) no human intervention. Differently, the model uses naturally existed supervision signals for training. In other words, the supervision signals exist in the unlabeled data and we utilize the prediction way (like supervised learning) to train the model.

**Highlight**

7. transfer learning: Train on one problem, but test on a different but related problem, e.g. multi-task learning, incremental learning and domain adaptation.
8. active learning: In the beginning, given little labels, the model should study which labeled samples are needed and meaningful. It aims to teach model to select more important data.
9. zero/one/few-shot learning: Given no/one/little samples, we try the best to train a classifier.

At last, reinforcement learning is different from deep learning. Based on the reward from environment, the model gives the reaction. Compared to self-supervised learning, the feedback is low. 

## Why do we need it?

No matter what CV or NLP with deep learning, representation learning is a fundamental problem. We usually use pretrain-then-finetune pipeline. As to this point, self-supervised learning can leverage self-labels for representation learning. Also, the representation contains high-level semantic information because of the human-like learning mode, which is meaningful for deep learning.

## How can we realize it?

### Context

### Video

### Cross-Modality

### Exemplar Learning
