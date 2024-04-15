nodes=(50 100 200 500)
densities=(2 4 8 16 24 32)
noises=(0 0.02 0.08 0.12 0.18 0.24 0.3 0.35)

for node in ${nodes[@]}; do
    for density in ${densities[@]}; do
        for noise in ${noises[@]}; do
            echo "Generating dataset ER[$node,$density,$noise]"

            rye run gnnco-generate gm \
                --output-dir "/scratch/jlagesse/gnnco/datasets/ER[$node,$density,$noise]" \
                --n-graphs 5000 \
                --n-val-graphs 500 \
                --order $node \
                --density $density \
                --noise $noise \
                --cuda
        done
    done
done