TRAIN_CONFIGURATIONS=("TEST_TRAIN" "TEST_TRAIN_DISTRIBUTED")

for configuration in "${TRAIN_CONFIGURATIONS[@]}"; do
  echo "Benchmarking current training run: $configuration"
  sbatch ./jpl_scripts/eval_benchmark_jpl.sh $configuration
done
