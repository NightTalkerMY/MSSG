#!/usr/bin/env bash
# Modification by William for Columbia Graph Evaluation

GPU=0
numv=2000
batch_size=64
lr=0.001
cp -r ./data/Columbia/graphs/graph_2000_0/train ./data/Columbia/graphs/graph_2000_0/val

# 1. Point to the Columbia graphs we just generated!
graph_path=./data/Columbia/graphs

echo "Booting up the Graph Neural Network..."

# 2. Run the Graph model
python train_graph.py --gpu_id=$GPU \
                      --numv=$numv \
                      --batch_size=$batch_size \
                      --graph_path=$graph_path \
                      --lr=$lr