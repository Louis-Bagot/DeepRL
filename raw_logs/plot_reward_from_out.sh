f=$1
csvf=${f%.*}.csv
grep 'episodic_return_train' $1 | cut --delimiter=" " -f9,11 | sed 's/ //g' > $csvf
python3 episodic_reward.py $csvf
