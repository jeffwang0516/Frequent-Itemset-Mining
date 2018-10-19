echo "Running Eclat CPU/GPU demo"
for minSup in 0.001 0.0008 0.0006 0.0004 0.0002
do
  bash eclat_cpu.sh dataset/data_large.txt ${minSup} sample_output/output_eclat_cpu_${minSup}.txt
  bash eclat_gpu.sh dataset/data_large.txt ${minSup} sample_output/output_eclat_gpu_${minSup}.txt
done