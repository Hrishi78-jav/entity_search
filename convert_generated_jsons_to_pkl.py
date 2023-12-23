import ast
import json
import os
from tqdm import tqdm
import pickle


def get_output2(gpt_output):
    output = ast.literal_eval(gpt_output['LLM_output'])
    result = []
    for val in output.values():
        if type(val) == list:
            result += val
    return result


def get_output(gpt_output):
    input = gpt_output['input'][0]['content'].split('input_product=')[1]  # see the format of output,input
    output = ast.literal_eval(gpt_output['output']['LLM_output'])
    result = []

    if type(output) == dict:
        for val in output.values():
            if type(val) == list:
                result += val
    elif type(output) == list:
        for d in output:
            if type(d) == dict:
                result += list(d.values())
            elif type(d) == list:
                result += d
            elif type(d) == str:
                result += [d]
    return input, result


if __name__ == '__main__':
    directory = 'C:/Users/hrishikesh/Desktop/negatives'  # change path
    file_list = [x for x in os.listdir(directory) if 'dec' in x]  # change the month
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[-1].replace('.json', '')))
    print(len(file_list))

    result = []
    count = 0
    for file_name in tqdm(file_list[:]):
        try:
            path = os.path.join(directory, file_name)
            with open(path, 'r') as file:
                data = json.load(file)
                input_word, output = get_output(data)
                result.append((str(input_word), str(output), file_name))
        except:
            result.append((str([]), file_name))
            count += 1
            print(file_name)

    with open('C:/Users/hrishikesh/Desktop/javis_product_negative_output.pkl', 'wb') as f:  #change output path
        pickle.dump(result, f)

    print(len(result), result[:5])
    print(f'failed = {count}')
