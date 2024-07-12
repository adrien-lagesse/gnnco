# nodes=(50 100 200 500)
# densities=(2 4 8 16 24 32)
noises=(0.02 0.08 0.12 0.18 0.24 0.3 0.35)


for noise in ${noises[@]}; do
    echo "Generating dataset CoraFull[$noise]"

    rye run gm-generate-corafull \
        -o "/home/jlagesse/gnnco/data/CoraFull[100,$noise]" \
        --n-graphs 2000 \
        --n-val-graphs 200 \
        --order 100 \
        --noise $noise 
done
