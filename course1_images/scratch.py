import os

print(os.path.join(os.path.dirname(os.getcwd()), 'data').replace('C:', 'D:'))
print(os.path.join(os.pardir, 'data', 'cats_and_dogs_filtered'))
