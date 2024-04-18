dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
dict3 = {'d': 5}

combined_dict = {}
combined_dict.update(dict1)
combined_dict.update(dict2)
combined_dict.update(dict3)

print(combined_dict)
