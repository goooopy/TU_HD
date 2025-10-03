
feature='node_attrlabels'
#feature='node_labeldegree'
for l in 2 1
do
output=proteins-$feature-$l
echo "" > $output
for seed in 42 101 102  203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset proteins --gpu 0 --n-layers $l --seed $seed  --use-feature $feature>>$output
done
grep neighbor $output  | grep Test | awk 'BEGIN{s=0} {s+=$6} END{print "MEAN",  s/5}' >>$output
done

#proteins-node_attrlabels-1:MEAN 0.72502
