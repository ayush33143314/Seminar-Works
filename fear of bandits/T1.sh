((i=0))
for instance in  "../instances/i-1.txt"
do
    for algorithm in  "epsGreedy" 
    do
        for horizon in  100 400 1600 6400 25600 102400
        do
            for seed in {0..49}
            do
                echo -ne "\\r $instance : $algorithm : $horizon : $seed : iteration $i                "
                python.exe bandit.py --instance $instance --algorithm $algorithm --randomSeed $seed --epsilon 0.02 --horizon $horizon >> outputDataT22.txt
                ((i=i+1))
            done
        done
    done
done