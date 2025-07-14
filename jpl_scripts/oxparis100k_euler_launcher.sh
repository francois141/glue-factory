for i in {0..15}; do
  echo "Launching chunk $i"
  sbatch ./jpl_scripts/oxparis100k_euler_worker.sh $i
done