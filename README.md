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

Compared to other machine learning paradigms:

**Polarization**

1. supervised learning: Given data and desired output, the model aims to finish the task.
2. unsupervised learning: No guidance at all. Only data with no label. We need to mine useful or meaningful information from the data distribution.

**In Between**

3. semi-supervised learning: 
