

for seed in 42 101 102  203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset mutag --gpu 0 --n-layers 3 --seed $seed  --use-feature node_labels
done
