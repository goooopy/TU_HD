
feature='nodeedge_labels'
l=12 #l=8 and l=10 gives mean 0.87, l=12 gives 0.88

for l in 1 2 4 6 8 12 16 18 20 
do
output=mutag-$feature-$l
echo "" > $output
for seed in 42 101 102  203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset mutag --gpu 0 --n-layers $l --seed $seed  --use-feature $feature>>$output
done
grep neighbor $output  | grep Test | awk 'BEGIN{s=0} {s+=$6} END {print "MEAN",  s/5}' >>$output
done

#mutag-nodeedge_labels-1:MEAN 0.83332
#mutag-nodeedge_labels-12:MEAN 0.87776
#mutag-nodeedge_labels-16:MEAN 0.88888
#mutag-nodeedge_labels-18:MEAN 0.88888
#mutag-nodeedge_labels-2:MEAN 0.86666
#mutag-nodeedge_labels-20:MEAN 0.88888
#mutag-nodeedge_labels-4:MEAN 0.87776
#mutag-nodeedge_labels-6:MEAN 0.85554
#mutag-nodeedge_labels-8:MEAN 0.86664

