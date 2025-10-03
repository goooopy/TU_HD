#python ./graph_classification_hebian_continuous1.py --dataset dd --gpu 0 --n-layers $l --seed $seed  --use-feature node_labels --alpha 1>>$output

feature='node_labels'
for alpha in 0.1 0.3 0.7 0.9
do
for l in  3
do
output=dd-$feature-$l-$alpha
echo "" > $output
for seed in 42 101 102 203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset dd --gpu 0 --n-layers $l --seed $seed  --use-feature $feature --alpha $alpha >>$output
done
grep -i multi_prototype $output  | awk 'BEGIN{s=0} {s+=$6} END {print " PROTO MEAN",  s/5}' >>$output
grep -i neighbor $output  | awk 'BEGIN{s=0} {s+=$6} END {print " NEIGHBOR MEAN",  s/5}' >>$output
done
done

#dd-node_labels-3: PROTO MEAN 0.64956
#dd-node_labels-3: NEIGHBOR MEAN 0.70926
#dd-node_labels-3-0.1: PROTO MEAN 0.65812
#dd-node_labels-3-0.1: NEIGHBOR MEAN 0.74118
#dd-node_labels-3-0.3: PROTO MEAN 0.66668
#dd-node_labels-3-0.3: NEIGHBOR MEAN 0.74286
#dd-node_labels-3-0.5: PROTO MEAN 0.6547
#dd-node_labels-3-0.5: NEIGHBOR MEAN 0.74958
#dd-node_labels-3-0.7: PROTO MEAN 0.64956
##dd-node_labels-3-0.7: NEIGHBOR MEAN 0.75294
#dd-node_labels-3-0.9: PROTO MEAN 0.65814
#dd-node_labels-3-0.9: NEIGHBOR MEAN 0.73782
