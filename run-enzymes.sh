
feature='node_attr'
for l in 1 2 3 4 6 8 12 16 18 20 
do
output=enzymes-$feature-$l
echo "" > $output
for seed in 42 101 102  203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset enzymes --gpu 0 --n-layers $l --seed $seed  --use-feature $feature>>$output
done
grep neighbor $output  | grep Test | awk 'BEGIN{s=0} {s+=$6} END{print "MEAN",  s/5}' >>$output
done

#enzymes-node_attr-1:MEAN 0.59334
#enzymes-node_attr-12:MEAN 0.51332
#enzymes-node_attr-16:MEAN 0.49
#enzymes-node_attr-18:MEAN 0.49
#enzymes-node_attr-2:MEAN 0.55666
#enzymes-node_attr-20:MEAN 0.48
#enzymes-node_attr-3:MEAN 0.57666
#enzymes-node_attr-4:MEAN 0.54668
#enzymes-node_attr-6:MEAN 0.53334
#enzymes-node_attr-8:MEAN 0.51334
