from common.inconsistents_parser import *
import os


filename = input('Filename: ')
incon_path = input('Inconsistent path: ')

input_data, output_data = inconsistents_parser(os.path.abspath(filename), os.path.abspath(incon_path))
print(input_data)
print(output_data)