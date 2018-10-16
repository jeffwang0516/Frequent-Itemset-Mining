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
            'naive_bit': 2,
            'compressed_bit': 3,
            'np_bit_array': 4
    }

    def __init__(self, use_data_struc='np_bit_array'):
        """
        Arg:
            use_data_struc = ['TIDSET', 'naive_bit', 'compressed_bit', 'np_bit_array']
        """
        super().__init__()
        self.tidset_data = dict()
        self.bitvector_data = dict()
        self.bitvector_data_compressed = dict()
        self.bitvector_data_with_numpy = dict()
        
        if use_data_struc not in self.DATA_STRUCT_AVAILABLE.keys():
            raise ValueError(
                "Data struc not available! Only ['TIDSET', 'naive_bit', 'compressed_bit', 'np_bit_array']  ")
        self.data_struc = self.DATA_STRUCT_AVAILABLE[use_data_struc]

        if self.data_struc == self.DATA_STRUCT_AVAILABLE['compressed_bit']:
            raise ValueError('Compressed bit still buggy! Please use other data struc!')

    def proccess_input_data(self, filename):
        print('Processing data input...')
        with open(filename, 'r') as fp:
            for line in fp:
                tx = [int(item) for item in line.split()]
                self.TXs_sets.append(frozenset(tx))
                for item in tx:
                    self.item_sets.add(frozenset([item]))

            for item in self.item_sets:
                if item not in self.tidset_data:
                    self.tidset_data[item] = set()
                if item not in self.bitvector_data:
                    self.bitvector_data[item] = list()  # ""
                if item not in self.bitvector_data_compressed:
                    self.bitvector_data_compressed[item] = list()
                
            # Create tidset & naive bit vector
            for idx, tx in enumerate(self.TXs_sets):
                for item in self.tidset_data.keys():
                    if item.issubset(tx):
                        self.tidset_data[item].update({idx+1})
                        self.bitvector_data[item].append(1)
                    else:
                        self.bitvector_data[item].append(0)

            if self.data_struc == self.DATA_STRUCT_AVAILABLE['np_bit_array']:
                # Create numpy bit array
                for item in self.bitvector_data.keys():
                    self.bitvector_data_with_numpy[item] = np.array(self.bitvector_data[item], dtype=np.int)
            
            elif self.data_struc == self.DATA_STRUCT_AVAILABLE['compressed_bit']:
                # Compressed bit vector data
                for item in self.item_sets:
                    raw_str = ''.join([str(bit)
                                    for bit in self.bitvector_data[item]])
                    first_bit = int(raw_str[0])
                    inv_bit = str(first_bit ^ 1)

                    try:
                        left_trailing_idx = raw_str.index(inv_bit)
                        right_trailing_idx = raw_str.rindex(inv_bit)+1

                        remain_frag = raw_str[left_trailing_idx:right_trailing_idx]

                        if first_bit == 1:
                            remain_frag = remain_frag.replace(
                                '0', 'x').replace('1', '0').replace('x', '1')

                        self.bitvector_data_compressed[item] += [
                            first_bit, left_trailing_idx, remain_frag]
                    except ValueError:
                        self.bitvector_data_compressed[item] += [
                            first_bit, len(raw_str), ""]

            self.TXs_amount = len(self.TXs_sets)

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
        else:
            # Operation with naive & compressed bit vector data
            for key, bit_vec in self.bitvector_data.items():
                self.support_dict[key] = sum(bit_vec)

    def matmul(self, A, B):
        return [x & y for x, y in zip(A, B)]

    def matmul_str(self, A, B):
        list_A = [int(c) for c in A]
        list_B = [int(c) for c in B]
        return ''.join([str(x & y) for x, y in zip(list_A, list_B)])

    def matmul_or_str(self, A, B):
        list_A = [int(c) for c in A]
        list_B = [int(c) for c in B]
        return ''.join([str(x | y) for x, y in zip(list_A, list_B)])

    def matmul_bitwise_complement_str(self, A):
        list_A = [int(c) for c in A]
        return ''.join([str(x ^ 1) for x in list_A])

    def compressed_bitvector_intersect(self, A, B):
        if A[1] > B[1]:
            A, B = B, A
        flag_A, rem_A, data_A = A
        flag_B, rem_B, data_B = B

        flag_res = 0
        rem_res = 0
        data_res = ""

        if flag_A == 0 and flag_B == 0:
            # 2 origin
            flag_res = 0

            rem_res = rem_B  # Larger one

            start_A = rem_B - rem_A
            end_A = len(data_A)

            overlap_len = end_A - start_A
            start_B = 0
            end_B = start_B + overlap_len

            # Find overlap part
            overlap_data_A = data_A[start_A:end_A]
            overlap_data_B = data_B[start_B:end_B]
            intersect = self.matmul_str(overlap_data_A, overlap_data_B)

            # Process output data
            left_trailing_idx = intersect.find('1')
            right_trailing_idx = intersect.rfind('1')+1

            if left_trailing_idx == -1:
                left_trailing_idx = 0
                right_trailing_idx = 0

            rem_res += left_trailing_idx
            data_res = intersect[left_trailing_idx:right_trailing_idx]

        elif flag_A == 1 and flag_B == 1:
            # 2 complement
            flag_res = 1

            if rem_A > rem_B:
                rem_A, data_A, rem_B, data_B = rem_B, data_B, rem_A, data_A
            # A is the smaller after exchanged
            rem_res = rem_A
            start_A = rem_B - rem_A
            end_A = len(data_A)

            overlap_len = end_A - start_A

            start_B = 0
            end_B = start_B + overlap_len

            # Find overlap part
            overlap_data_A = data_A[start_A:end_A]
            overlap_data_B = data_B[start_B:end_B]
            intersect = self.matmul_or_str(overlap_data_A, overlap_data_B)

            tail = ''
            if len(intersect) < overlap_len:
                tail = overlap_data_A[len(intersect):]
            elif end_B <= len(data_B):
                tail = data_B[end_B:]

            data_res = data_A[:start_A] + intersect + tail

            # Check to output original or complement
            if rem_A == 0 or rem_B == 0 or len(data_A)+len(data_B)+rem_A+rem_B >= self.TXs_amount:
                flag_res = 0
                data_comp = self.matmul_bitwise_complement_str(data_res)
                left_trailing_idx = data_comp.find('1')
                right_trailing_idx = data_comp.rfind('1')+1

                if left_trailing_idx == -1:
                    rem_res = self.TXs_amount
                    data_res = ''
                else:
                    rem_res = left_trailing_idx
                    data_res = data_comp[left_trailing_idx:right_trailing_idx]

        else:
            # Always output 0
            flag_res = 0

            if flag_B == 0:
                # B origin A complement, exchange AB first
                # print('Exchanged')
                rem_A, data_A, rem_B, data_B = rem_B, data_B, rem_A, data_A

            # A after exchanged is always original
            # A origin B complement
            rem_res = rem_A  # choose one with original

            if rem_A > rem_B:
                start_B = rem_A - rem_B
                end_B = len(data_B)

                overlap_len = end_B - start_B

                start_A = 0
                end_A = start_A + overlap_len
                # Find overlap part
                overlap_data_A = data_A[start_A:end_A]
                overlap_data_B = data_B[start_B:end_B].replace(
                    '0', 'x').replace('1', '0').replace('x', '1')
                intersect = self.matmul_str(overlap_data_A, overlap_data_B)
                # print((overlap_data_A, overlap_data_B), intersect)

                # Process output data
                left_trailing_idx = intersect.find('1')
                # right_trailing_idx = intersect.rfind('1')+1

                if left_trailing_idx == -1:
                    left_trailing_idx = 0
                    itersect_result = ""
                    # right_trailing_idx = 0
                else:
                    itersect_result = intersect[left_trailing_idx:]

                rem_res += left_trailing_idx
                data_res = itersect_result + data_A[end_A:]
            else:
                start_Ａ = rem_B - rem_A
                end_A = len(data_A)

                overlap_len = end_A - start_A

                start_B = 0
                end_B = start_B + overlap_len

                # Find overlap part
                overlap_data_A = data_A[start_A:end_A]
                overlap_data_B = data_B[start_B:end_B].replace(
                    '0', 'x').replace('1', '0').replace('x', '1')
                intersect = self.matmul_str(overlap_data_A, overlap_data_B)
                # print((overlap_data_A, overlap_data_B), intersect)

                # Process output data
                # left_trailing_idx = intersect.find('1')
                right_trailing_idx = intersect.rfind('1')+1

                if right_trailing_idx == 0:
                    itersect_result = ""
                    # right_trailing_idx = 0
                else:
                    itersect_result = intersect[:right_trailing_idx]

                # rem_res += left_trailing_idx

                data_res = data_A[:start_A] + itersect_result

            # else:
            #     # B origin A complement

            #     rem_res = rem_B # choose one with original
            #     pass

        return [flag_res, rem_res, data_res]

    def refresh_support(self, dict_to_check=None):
        if self.data_struc == self.DATA_STRUCT_AVAILABLE['TIDSET']:
            # Operation with Tidset data
            for key in dict_to_check.keys():
                tidsets = [self.tidset_data[frozenset([item])] for item in key]
                intersect = set.intersection(*tidsets)
                dict_to_check[key] = len(intersect)

        elif self.data_struc == self.DATA_STRUCT_AVAILABLE['naive_bit']:

            # Operation with naive bit vector data
            for key in dict_to_check.keys():
                bitvec_sets = [
                    self.bitvector_data[frozenset([item])] for item in key]

                res = [1 for i in range(len(bitvec_sets[0]))]

                for i in range(0, len(bitvec_sets), 2):
                    A = bitvec_sets[i]
                    if i+1 >= len(bitvec_sets):
                        result_AB = A
                    else:
                        B = bitvec_sets[i+1]
                        result_AB = self.matmul(A, B)

                    res = self.matmul(res, result_AB)

                dict_to_check[key] = sum(res)

        elif self.data_struc == self.DATA_STRUCT_AVAILABLE['compressed_bit']:
            # Operation with compressed bit vector data
            for key in dict_to_check.keys():
                bitvec_sets = [
                    self.bitvector_data_compressed[frozenset([item])] for item in key]
                res = [1, self.TXs_amount, '']

                for i in range(0, len(bitvec_sets), 2):
                    A = bitvec_sets[i]
                    if i+1 >= len(bitvec_sets):
                        result_AB = A
                    else:
                        B = bitvec_sets[i+1]
                        result_AB = self.compressed_bitvector_intersect(A, B)

                    res = self.compressed_bitvector_intersect(res, result_AB)

                num_of_1s = sum([int(c) for c in res[2]])
                if res[0] == 0:
                    dict_to_check[key] = num_of_1s
                else:
                    dict_to_check[key] = self.TXs_amount - num_of_1s

        elif self.data_struc == self.DATA_STRUCT_AVAILABLE['np_bit_array']:
            # Operation with numpy bit array
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
