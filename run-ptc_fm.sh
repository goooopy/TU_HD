
feature='nodeedge_labels'
for l in 1 2 3
do
output=ptc_fm-$feature-$l
echo "" > $output
for seed in 42 101 102 203 432 
do
python ./graph_classification_hebian_continuous1.py --dataset ptc_fm --gpu 0 --n-layers $l --seed $seed  --use-feature $feature>>$output
done
grep -i multi_prototype $output  | awk 'BEGIN{s=0} {s+=$6} END {print " PROTO MEAN",  s/5}' >>$output
grep -i neighbor $output  | awk 'BEGIN{s=0} {s+=$6} END {print " NEIGHBOR MEAN",  s/5}' >>$output
done

# ptc_fm-nodeedge_labels-1: NEIGHBOR MEAN 0.60002
