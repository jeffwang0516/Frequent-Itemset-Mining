import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


class FrequentItemsetAlgorithm():
    """Base class"""

    def __init__(self):
        self.support_dict = {}  # records all frequent itemset
        self.TXs_amount = 0
        self.TXs_sets = []
        self.item_sets = set()

    def proccess_input_data(self, filename):
        pass

    def save_output(self, filename):
        print("Saving output to file:", filename)
        # supportList = sorted(self.support_dict.items(), key=lambda t: len(t[0]))
        with open(filename, 'w') as fp:
            for k, v in self.support_dict.items():
                items = ' '.join([str(item) for item in sorted(k)])
                fp.write('{} ({})\n'.format(items, v))

    def calculate_one_item_support(self):
        pass

    def refresh_support(self, dict_to_check=None):
        pass

    def check_minimum_support(self, min_support_ratio, dict_to_check=None):
        to_be_pruned = []
        if dict_to_check is None:
            dict_to_check = self.support_dict

        for k, v in dict_to_check.items():
            if v / float(self.TXs_amount) < min_support_ratio:
                to_be_pruned.append(k)

        for k in to_be_pruned:
            dict_to_check.pop(k)

    def find_supersets_k(self, min_support_ratio):
        pass


class Apriori(FrequentItemsetAlgorithm):
    def __init__(self):
        super().__init__()

    def proccess_input_data(self, filename):
        print('Processing data input...')
        with open(filename, 'r') as fp:
            for line in fp:
                tx = [int(item) for item in line.split()]
                self.TXs_sets.append(frozenset(tx))

                for item in tx:
                    self.item_sets.add(frozenset([item]))
            self.TXs_amount = len(self.TXs_sets)

    def calculate_one_item_support(self):
        self.support_dict = {}
        for tx in self.TXs_sets:
            for item in self.item_sets:
                if item.issubset(tx):
                    try:
                        self.support_dict[item] += 1
                    except KeyError:
                        self.support_dict[item] = 1

    def refresh_support(self, dict_to_check=None):
        if not dict_to_check or type(dict_to_check) != dict:
            return
        for tx in self.TXs_sets:
            for k in dict_to_check.keys():
                if k.issubset(tx):
                    dict_to_check[k] += 1

    def find_supersets_k(self, min_support_ratio):
        # Start from Level 1 (One item)
        self.calculate_one_item_support()
        self.check_minimum_support(min_support_ratio)

        # Iterates to higher k+1 level, till None
        previous_level_dict = self.support_dict

        while True:
            print("> Finding next level superset from", len(
                previous_level_dict.items()), "k-1 items..")

            # New dict for this new level
            current_level_dict = dict()

            items = list(previous_level_dict.items())

            item_sets = set()
            for item in previous_level_dict.keys():
                item_sets.add(item)

            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    k1, v1 = items[i]
                    k2, v2 = items[j]

                    new_key = k1 | k2

                    # Below checks if the new_key's length matches with the current level
                    continue_flag = 0
                    for k in new_key:
                        temp_set = frozenset(new_key - set({k}))
                        test_set = frozenset([temp_set])

                        if not test_set.issubset(item_sets):
                            continue_flag = 1
                            break

                    if new_key not in self.support_dict and continue_flag == 0:
                        current_level_dict[new_key] = 0

            self.refresh_support(current_level_dict)
            self.check_minimum_support(min_support_ratio, current_level_dict)

            if len(current_level_dict.items()) == 0:
                print("Frequent Itemsets generation finished !!!")
                return

            # Take this k level output to generate k+1 level
            previous_level_dict = current_level_dict

            # Update into support_dict, records all frequent itemset
            self.support_dict.update(current_level_dict)

class Eclat(FrequentItemsetAlgorithm):
    DATA_STRUCT_AVAILABLE = {
            'TIDSET': 1,
            # 'naive_bit': 2,
            # 'compressed_bit': 3,
            'np_bit_array': 4
    }
    
    GPU_MOD = SourceModule("""
        __device__ void warpReduce(volatile int* sdata, int tid) {
            sdata[tid] += sdata[tid + 32];
            sdata[tid] += sdata[tid + 16];
            sdata[tid] += sdata[tid + 8];
            sdata[tid] += sdata[tid + 4];
            sdata[tid] += sdata[tid + 2];
            sdata[tid] += sdata[tid + 1];
        }
        __global__ void count_support(char *g_idata, int *g_odata, int *DATA_SIZE) {//(char *bit_array, int *itemset_id, int *ITEM_SIZE, int *g_odata, int *DATA_SIZE) {

            extern __shared__ int sdata[];

            unsigned int tid = threadIdx.x;
            unsigned int bid = blockIdx.x;
            
            
            sdata[tid] = 0;
            for (unsigned int i = bid * (blockDim.x) + tid; i< (*DATA_SIZE); i+=gridDim.x*blockDim.x){
                sdata[tid] += g_idata[i]; 
                __syncthreads();
            }

            __syncthreads();

            for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            if (tid < 32) warpReduce(sdata, tid);

            // write result for this block to global mem
            if (tid == 0) {
                g_odata[blockIdx.x] = sdata[0];
            }

        }
        __global__ void op_and(char *bit_array, char *op_and_odata, int *itemset_id, int *ITEM_SIZE, int *DATA_SIZE)
        {
            /*
                bit_array
                
                01000...1100.. [1st]
                10010...0111.. [2nd]
                11011...1101.. [3nd]
                ..
                ..
            */
            
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int offset0 = itemset_id[0]*(*DATA_SIZE);
            for (int i = index; i < *DATA_SIZE; i += stride){
                op_and_odata[i] = bit_array[i + offset0];
                if(!op_and_odata[i]) continue;
                for (int j = 1; j < *ITEM_SIZE; j+=1) {
                    op_and_odata[i] = op_and_odata[i] & bit_array[i + itemset_id[j]*(*DATA_SIZE)];
                }
            }
        }
      """)

    def __init__(self, use_data_struc='np_bit_array', use_gpu=False, block_param_opAND=(512,1,1), grid_param_opAND=(128,1), block_param_sum=(512,1,1), grid_param_sum=(128,1)):
        """
        Arg:
            use_data_struc = ['TIDSET', 'np_bit_array']
        """
        super().__init__()
        self.tidset_data = dict()
        self.bitvector_data_with_numpy = dict()
        self.use_gpu = use_gpu

        if use_gpu:
            # block, grid param for two gpu functions
            self.block_param_opAND = block_param_opAND
            self.grid_param_opAND = grid_param_opAND
            self.block_param_sum = block_param_sum
            self.grid_param_sum = grid_param_sum
            
            self.data_gpu = None
            self.shared_mem = None
            self.result = None
            self.result_gpu = None
            self.num_of_TXs_gpu = None
            self.op_and_func = self.GPU_MOD.get_function("op_and")
            self.count_support = self.GPU_MOD.get_function("count_support")
            
            self.bit_2DArray_gpu = None
            self.bit_vec_keys_gpu = None
            self.itemset_size_gpu = None

        if use_data_struc not in self.DATA_STRUCT_AVAILABLE.keys():
            raise ValueError(
                "Data struc not available! Only ['TIDSET', 'np_bit_array']  ")
        self.data_struc = self.DATA_STRUCT_AVAILABLE[use_data_struc]

    def proccess_input_data(self, filename):
        print('Processing data input...')
        with open(filename, 'r') as fp:
            txCnt = 0
            for idx, line in enumerate(fp):
                tx = [int(item) for item in line.split()]
                self.TXs_sets.append(frozenset(tx))
                for item in tx:
                    self.item_sets.add(frozenset([item]))
                    key = frozenset([item])
                    if key in self.tidset_data.keys():
                        self.tidset_data[key].update({txCnt})
                    else:
                        self.tidset_data[key] = set({txCnt})
                txCnt += 1
            
            # Create numpy bit array
            print('Creating bit vector from TIDSET...')
            for item in self.tidset_data.keys():
                bit_array = np.zeros(txCnt, dtype=np.int8)
                for tid in self.tidset_data[item]:
                    bit_array[tid] = np.int8(1)
                self.bitvector_data_with_numpy[item] = bit_array
            
            self.TXs_amount = txCnt
            
            # Initialize data for GPU processing
            # memcpy is timeconsuming, do it first here
            if self.use_gpu:
                print('Initializing GPU MEM')
                
                # Store output of the first func op_and and use as input to count_support
                intermediate_data = np.zeros([txCnt], dtype=np.int8)
                self.intermediate_data_gpu = cuda.mem_alloc(intermediate_data.nbytes)
                
                # Final output
                self.result = np.zeros([self.block_param_sum[0]], dtype=np.int32)
                self.result_gpu = cuda.mem_alloc(self.result.nbytes)
                cuda.memcpy_htod(self.result_gpu, self.result)

                # Store Bit vector length = TX amount
                num_of_TXs = np.int32(txCnt)
                self.num_of_TXs_gpu = cuda.mem_alloc(num_of_TXs.nbytes)
                cuda.memcpy_htod(self.num_of_TXs_gpu, num_of_TXs)

                # GPU func count_support needs shared mem (4 * thread_size)
                self.shared_mem = 4* self.block_param_sum[0]

                # Store the keys of bit vector to do intersection func on GPU
                bit_vec_keys = np.arange(2603, dtype=np.int32)
                self.bit_vec_keys_gpu = cuda.mem_alloc(bit_vec_keys.nbytes)
                bit_vec_keys_size = np.int32(2603)
                self.bit_vec_keys_size_gpu = cuda.mem_alloc(bit_vec_keys_size.nbytes)
                
                # Copy All Bit vector into gpu
                bit_2DArray_key_val = sorted(self.bitvector_data_with_numpy.items(), key=lambda x: tuple(x[0]))
                
                bit_2DArray = []
                for item in bit_2DArray_key_val:
                    bit_2DArray.append(item[1])
                bit_2DArray = np.array(bit_2DArray, dtype=np.uint8)
                self.bit_2DArray_gpu = cuda.mem_alloc(bit_2DArray.nbytes)
                cuda.memcpy_htod(self.bit_2DArray_gpu, bit_2DArray)
                print("Whole bit vector copied to GPU Mem! shape:",bit_2DArray.shape, "size:", bit_2DArray.nbytes, "bytes")

    def calculate_one_item_support(self):
        self.support_dict = dict()

        if self.data_struc == self.DATA_STRUCT_AVAILABLE['TIDSET']:
            # Operation with Tidset data
            for key, tidset in self.tidset_data.items():
                self.support_dict[key] = len(tidset)
        elif self.data_struc == self.DATA_STRUCT_AVAILABLE['np_bit_array']:
            # Operation with numpy bit array
            for key, bit_array in self.bitvector_data_with_numpy.items():
                self.support_dict[key] = np.sum(bit_array)

    def refresh_support(self, dict_to_check=None):
        if self.data_struc == self.DATA_STRUCT_AVAILABLE['TIDSET']:
            # Operation with Tidset data
            for key in dict_to_check.keys():
                tidsets = [self.tidset_data[frozenset([item])] for item in key]
                intersect = set.intersection(*tidsets)
                dict_to_check[key] = len(intersect)

        elif self.data_struc == self.DATA_STRUCT_AVAILABLE['np_bit_array']:
            # Operation with numpy bit array
            if self.use_gpu is True:
                # Use gpu to do intersection & count support
                intersect_sets = []
                intersect_sets_key = []
                for key in dict_to_check.keys():
                    # Find what keys to do intersections on their bitvectors
                    bit_vec_keys = np.array([int(item)-1 for item in key], dtype=np.int32)
                    cuda.memcpy_htod(self.bit_vec_keys_gpu, bit_vec_keys)
                    
                    bit_vec_keys_size = np.int32(len(bit_vec_keys))
                    cuda.memcpy_htod(self.bit_vec_keys_size_gpu, bit_vec_keys_size)
    
                    # Do Intersection first
                    self.op_and_func(self.bit_2DArray_gpu, self.intermediate_data_gpu, self.bit_vec_keys_gpu, self.bit_vec_keys_size_gpu, self.num_of_TXs_gpu, block=self.block_param_opAND, grid=self.grid_param_opAND)
                    
                    # Then Count support                
                    self.count_support(self.intermediate_data_gpu, self.result_gpu, self.num_of_TXs_gpu, block=self.block_param_sum, grid=self.grid_param_sum, shared=self.shared_mem)
                    cuda.memcpy_dtoh(self.result, self.result_gpu)
                    
                    dict_to_check[key] = np.sum(self.result)

            else:
                for key in dict_to_check.keys():
                    bitvec_sets = [
                        self.bitvector_data_with_numpy[frozenset([item])] for item in key]

                    res = bitvec_sets[0]
                    for i in range(1, len(bitvec_sets)):
                        res = res*bitvec_sets[i]
                    
                    dict_to_check[key] = np.sum(res)

    def find_supersets_k(self, min_support_ratio):
        # Start from Level 1 (One item)
        self.calculate_one_item_support()
        self.check_minimum_support(min_support_ratio)

        # Iterates to higher k+1 level, till None
        previous_itemsets_of_prefix = {0: self.support_dict}

        while True:
            print("> Finding next level superset from", len(
                previous_itemsets_of_prefix.items()), "same prefix itemsets")

            # New dict for this new level
            current_itemsets_of_prefix = dict()
            num_of_super_k_sets_found_in_all_tree = 0
            for _, prev_prefix_dict in previous_itemsets_of_prefix.items():
                items = list(prev_prefix_dict.items())

                for i in range(len(items)):
                    k1, v1 = items[i]
                    current_prefix_dict = dict()

                    for j in range(i+1, len(items)):

                        k2, v2 = items[j]
                        new_key = k1 | k2

                        if new_key not in self.support_dict:
                            current_prefix_dict[new_key] = 0

                    self.refresh_support(current_prefix_dict)
                    self.check_minimum_support(
                        min_support_ratio, current_prefix_dict)
                    current_itemsets_of_prefix[k1] = current_prefix_dict

                    num_of_super_k_sets_found_in_all_tree += len(
                        current_prefix_dict.items())

            if num_of_super_k_sets_found_in_all_tree == 0:
                print("Frequent Itemsets generation finished !!!")
                return

            # Take this k level output to generate k+1 level
            previous_itemsets_of_prefix = current_itemsets_of_prefix

            # Update into support_dict, records all frequent itemset
            for _, current_prefix_dict in current_itemsets_of_prefix.items():
                self.support_dict.update(current_prefix_dict)
