# import os
# import csv
import numpy as np

# fname_path = os.path.join(os.path.dirname(os.getcwd()),'data','kaggle_sign_language','sign_mnist_test.csv')
# print(fname_path)
# with open(fname_path,'r') as f_obj:
#     csv_reader = csv.reader(f_obj)
#     i =0
#     for row in csv_reader:
#         print(row)
#         print('-------------------------------------------------------------')
#         i += 1
#         if i > 5:
#             break

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 87])
print(len(x))
