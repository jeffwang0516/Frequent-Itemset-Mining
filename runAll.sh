echo "Running Apriori demo"
for minSup in 0.35 0.30 0.25 0.20 
do
  bash apriori.sh dataset/data.txt ${minSup} sample_output/output_apriori_${minSup}.txt
done

echo "Running Eclat demo"
for minSup in 0.35 0.30 0.25 0.20 0.15 0.10 0.05
do
  bash eclat.sh dataset/data.txt ${minSup} sample_output/output_eclat_${minSup}.txt
done