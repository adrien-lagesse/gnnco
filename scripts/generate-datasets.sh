# nodes=(50 100 200 500)
# densities=(2 4 8 16 24 32)
noises=(0 0.02 0.08 0.12 0.18 0.24 0.3 0.35)


for noise in ${noises[@]}; do
    echo "Generating dataset ER[100,8,$noise]"

    rye run gm-generate-aqsol \
        -o "/home/jlagesse/gnnco/data/AQSOL[$noise]" \
        --noise $noise 
done
