
#output=nci1
#l=3 
#echo "" > $output
#for seed in 42 101 102  203 432 
#do
#python ./graph_classification_hebian_continuous1.py --dataset nci1 --gpu 0 --n-layers $l --seed $seed  --use-feature node_labels >>$output
#done



feature='node_labeldegree'
for alpha in 0.05 0.5 0.95
do
for l in 32 24
do
output=nci1-$feature-$l-$alpha
echo "" > $output
for seed in 42 101 102  203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset nci1 --gpu 0 --n-layers $l --seed $seed  --use-feature $feature --alpha $alpha >>$output
done
grep neighbor $output  | grep Test | awk 'BEGIN{s=0} {s+=$6} END{print "MEAN",  s/5}' >>$output
done
done

#nci1-node_labeldegree-32-0.05:MEAN 0.72554
#nci1-node_labeldegree-32-0.5:MEAN 0.72312

