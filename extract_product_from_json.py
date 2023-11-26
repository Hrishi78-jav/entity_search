import ast
import json
import os
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    directory = 'D:\Javis_Projects\product_data'  # change path
    folder_list = [x for x in os.listdir(directory)]  # change the month
    print(len(folder_list))

    result = {}
    count = 0
    for folder_name in tqdm(folder_list[:]):
        folder_path = os.path.join(directory, folder_name)
        file_path = folder_path + f'\{folder_name}.json'
        print(file_path)
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                d = json.loads(line)
                if 'title' in d:
                    data.append(d['title'])
        result[folder_name] = str(data)
        print(len(data), data[:3])

    with open('C:/Users/hrishikesh/Desktop/amazon_product_domain_wise.pkl', 'wb') as f:
        pickle.dump(result, f)
    #
    # print(len(result), result[:5])
    # print(f'failed = {count}')
