
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
    def __init__(self):
        super().__init__()
        self.bitvector_data = dict()

    def proccess_input_data(self, filename):
        with open(filename, 'r') as fp:
            for line in fp:
                tx = [int(item) for item in line.split()]
                self.TXs_sets.append(frozenset(tx))
                for item in tx:
                    self.item_sets.add(frozenset([item]))

            for item in self.item_sets:
                if item not in self.bitvector_data:
                    self.bitvector_data[item] = set()

            # Create bit vector
            for idx, tx in enumerate(self.TXs_sets):
                for item in self.bitvector_data.keys():
                    if item.issubset(tx):
                        self.bitvector_data[item].update({idx+1})

            self.TXs_amount = len(self.TXs_sets)

    def calculate_one_item_support(self):
        self.support_dict = dict()
        for key, tidset in self.bitvector_data.items():
            self.support_dict[key] = len(tidset)

    def refresh_support(self, dict_to_check=None):
        for key in dict_to_check.keys():

            tidsets = [self.bitvector_data[frozenset([item])] for item in key]
            intersect = set.intersection(*tidsets)
            dict_to_check[key] = len(intersect)

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
