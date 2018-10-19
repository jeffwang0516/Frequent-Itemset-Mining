import sys
import time

from association_rules.frequent_itemset_mining import Eclat

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Please use correct args!")
        sys.exit()
    input_file = sys.argv[1]
    min_support_ratio = float(sys.argv[2])
    output_file = sys.argv[3]

    eclat = Eclat(
        use_data_struc='np_bit_array',
        use_gpu=False
    )

    # Preprocess data
    eclat.proccess_input_data(input_file)

    print("[Eclat-CPU] Finding freq Itemset with min support >",
          min_support_ratio, '--')
    start_time = time.time()

    eclat.find_supersets_k(min_support_ratio)
    eclat.save_output(output_file)

    end_time = time.time()

    print("Elapsed time =", end_time-start_time)
    print('----------\n')
