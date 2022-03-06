# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="wtb5ZO5djGKi"
# # Small data and deep learning
# This pratical session proposes to study several techniques to improve training performance in the challenging context where few data and resources are available.

# %% [markdown] id="nkEk9hbxjGKk"
# # Introduction
# Assume we are in a context where few "gold" labeled data are available for training, say $\mathcal{X}_{\text{train}}\triangleq\{(x_n,y_n)\}_{n\leq N_{\text{train}}}$, where $N_{\text{train}}$ is small. A large test set $\mathcal{X}_{\text{test}}$ is available. A large amount of unlabeled data, $\mathcal{X}_{\text{nolabel}}$, is available. We also assume that we have a limited computational budget (e.g., no GPUs).
#
# For each question, write a commented *Code* or a complete answer as a *Markdown*. When the objective of a question is to report a CNN accuracy, please use the following format to report it, at the end of the question :
#
#
# | Model | Number of  epochs  | Train accuracy | Test accuracy |
# |------|------|------|------|
# |   XXX  | XXX | XXX | XXX |
#
#
# If applicable, please add the field corresponding to the  __Accuracy on Full Data__ as well as a link to the __Reference paper__ you used to report those numbers. (You do not need to train a CNN on the full CIFAR10 dataset.)
#
# In your final report, please keep the logs of each training procedure you used. We will only run this jupyter if we have some doubts on your implementation. 
#
# __The total file sizes should be reasonable (feasible with 2MB only!). You will be asked to hand in the notebook, together with any necessary files required to run it if any.__
#
#
# To run your experiments, you can use the same local installation as for previous TPs, or otherwise https://colab.research.google.com/.

# %% [markdown] id="yyz4hH-GjGKk"
# ## Training set creation
# __Question 1 (2 points) :__ Propose a dataloader that will only use the first 100 samples of the CIFAR-10 training set. 
#
# *Hint* : You can modify the file located at https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py or use the information from https://pytorch.org/vision/stable/datasets.html

# %% [markdown] id="Sx8Y526AjGKl"
# This is our dataset $\mathcal{X}_{\text{train}}$, it will be used until the end of this project. The remaining samples correspond to $\mathcal{X}_{\text{nolabel}}$. The testing set $\mathcal{X}_{\text{test}}$ corresponds to the whole testing set of CIFAR-10.

# %% id="SCeaqjCbjGKl"

# %% [markdown] id="kcFD3xMejGKm"
# ## Testing procedure
# __Question 2 (1.5 points) :__ Explain why the evaluation of the training procedure is difficult. Propose several solutions.

# %% [markdown] id="o7ZfGMuRjGKm"
#

# %% [markdown] id="b8yeRD7CjGKm"
# # Raw approach: the baseline

# %% [markdown] id="4dgKvuS0jGKm"
# In this section, the goal is to train a CNN on $\mathcal{X}_{\text{train}}$ and compare its performance with reported numbers from the litterature. You will have to re-use and/or design a standard classification pipeline. You should optimize your pipeline to obtain the best performances (image size, data augmentation by flip, ...).
#
# The key ingredients for training a CNN are the batch size, as well as the learning rate schedule, i.e. how to decrease the learning rate as a function of the number of epochs. A possible schedule is to start the learning rate at 0.1 and decreasing it every 30 epochs by 10. In case of divergence, reduce the learning rate. A potential batch size could be 10, yet this can be cross-validated.
#
# You can get some baselines accuracies in this paper : http://openaccess.thecvf.com/content_cvpr_2018/papers/Keshari_Learning_Structure_and_CVPR_2018_paper.pdf. Obviously, it is a different context for those researchers who had access to GPUs.

# %% [markdown] id="pJpkBpyWjGKn"
# ## ResNet architectures

# %% [markdown] id="z22FjUNYjGKn"
# __Question 3 (4 points) :__ Write a classification pipeline for $\mathcal{X}_{\text{train}}$, train from scratch and evaluate a *ResNet-18* architecture specific to CIFAR10 (details about the ResNet-18 model, originally designed for the ImageNet dataset, can be found here: https://arxiv.org/abs/1512.03385). Please report the accuracy obtained on the whole dataset as well as the reference paper/GitHub link.
#
# *Hint:* You can re-use the following code: https://github.com/kuangliu/pytorch-cifar. During a training of 10 epochs, a batch size of 10 and a learning rate of 0.01, one obtains 40% accuracy on $\mathcal{X}_{\text{train}}$ (\~2 minutes) and 20% accuracy on $\mathcal{X}_{\text{test}}$ (\~5 minutes).

# %% id="h9wmxgUSjGKn"

# %% [markdown] id="JLv78we5jGKo"
# # Transfer learning

# %% [markdown] id="et0xds0xjGKo"
# We propose to use pre-trained models on a classification task, in order to improve the results of our setting.

# %% [markdown] id="ErqP4xNMjGKo"
# ## ImageNet features

# %% [markdown] id="CyHaYiMCjGKo"
# Now, we will use a model pre-trained on ImageNet and see how well it performs on CIFAR. A list of ImageNet pre-trained models is available on : https://pytorch.org/vision/stable/models.html
#
# __Question 4 (3 points) :__ Pick a model from the list above, adapt it for CIFAR and retrain its final layer (or a block of layers, depending on the resources to which you have access to). Report its accuracy.

# %% id="DQu_qudYjGKo" tags=[]

# %% [markdown] id="-svmb56SjGKp"
# # Incorporating priors
# Geometrical priors are appealing for image classification tasks. 
# A color image $x$ can be seen as a function: $\mathbb{S}\rightarrow\mathbb{R}^3$, where $\mathbb{S} \subset \mathbb{R}^2$ is the image support.
# Let us consider transformations $\mathcal{T}$ of possible inputs $x$. For instance, if an image had infinite support, a translation $\mathcal{T}_a$ of an image $x$ by a shift $a$ would lead to a new infinite-support image $\mathcal{T}_a(x)$, described at each pixel $u$ by :
#
# $$\forall u, \mathcal{T}_a(x)(u)=x(u-a)\,.$$
#
# __Question 5 (1.5 points) :__ Explain the issues when dealing with translations, rotations, scaling effects, color changes on $32\times32$ images. Propose several ideas to tackle them.

# %% [markdown] id="9YJCU9PgjGKp"
#

# %% [markdown] id="CxMnR6QNjGKp"
# ## Data augmentations

# %% [markdown] id="vvohBWdmjGKp"
# __Question 6 (3 points) :__ Propose a set of geometric transformations beyond translation, and incorporate them in your training pipeline. Train the model of the __Question 3__ with them and report the accuracies.
# You can use tools from https://pytorch.org/vision/stable/transforms.html 

# %% id="PXA1-KP2jGKp" tags=[]

# %% [markdown] id="1I81WSECjGKq"
# # Conclusions

# %% [markdown] id="xjQRkH-kjGKq"
# __Question 7 (5 points) :__ Write a short report explaining the pros and the cons of each method you implemented. 25% of the grade of this project will correspond to this question, thus, it should be done carefully. In particular, please add a plot that will summarize all your numerical results.

# %% [markdown] id="82dUc-f5jGKq"
#

# %% [markdown] id="8MIfKUUojGKq"
# # Weak supervision

# %% [markdown] id="RldQ1hdPjGKq"
# __Bonus \[open\] question (up to 4 points) :__ Pick a weakly supervised method that will potentially use $\mathcal{X}_{\text{nolabel}}\cup\mathcal{X}_{\text{train}}$ to train a model (a subset of $\mathcal{X}_{\text{nolabel}}$ is also fine). Evaluate it and report the accuracies. You should be careful in the choice of your method, in order to avoid heavy computational effort.

# %% [markdown] id="xk1FF9bqjGKq"
#
