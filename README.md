# Paper Categories Predication Based on GNN
## How to run?
1. Install Pytorch, dgl(https://www.dgl.ai/), numpy and scikit-learn.
2. `python train.py --gpu 0 --dataset cora` or `python train.py --gpu 0 --dataset computer_amazon`.
## I. INTRODUCTION
&emsp;&emsp;For researchers, they often have many 
papers to read, and it is hard and time consuming for them to manage them manually. 
It will help save them much time if they have a 
tool to classify papers according to the words in 
papers. We are interested in implementing a 
model which can classify the papers into 
different topic categories. Also, it can be useful 
for some academic websites like Google 
scholar. After researchers click a certain paper, 
with this model, the website can try to 
recommend those papers belonging to the same 
category (Admittedly, it is much more 
complicated to implement a recommendation 
system in a real production system and may 
require more other recommendation 
algorithms). We decide to implement a model 
to predict paper subject categories on a large 
citation dataset.
## II. DATA
&emsp;&emsp;Our dataset is Cora. It is a graph-structured 
dataset. Each node represents a paper, and each 
edge represents a citation from the source paper 
to the target paper.&nbsp;

&emsp;&emsp;It is a famous dataset that has a total of 2708 
research papers and all samples belong to 7 
categories. Each sample paper is represented by 
a 1433-dimensional feature vector. Feature 
vectors are constructed by the bag-of-words model.&nbsp;
<img width="312" alt="image" src="https://user-images.githubusercontent.com/91409788/179866214-3c001f08-1aeb-408b-aeed-3e4ccf8e584c.png">
## III. MODEL - GraphSage

## IV. Techniques to avoid overfitting
### 1. Early Stopping
### 2. Dropout
