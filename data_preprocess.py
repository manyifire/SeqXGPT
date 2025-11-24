import json
import pandas as pd
from pathlib import Path
from pprint import pprint
from openai import OpenAI
import random
import re

def load_data(file_path):
    """
    Load data from a JSONl file.
    return: list of dict
    """
    all_files = []
    directory = Path(file_path)

    for json_file in directory.glob('*'):
        with open(json_file, 'r') as f:
            all_files.extend(f.readlines())
    
    return all_files

def extract_code(data, min_length=150):
    """
    Extract code from the data where the length of code tokens is greater than 150.
    return: list of dict
    data: list of dict
    """
    code_data = []
    for item in data:
        item = json.loads(item)
        if len(item['code_tokens']) > min_length:
            code_data.append(item)
    return code_data

def get_code_completion(prompt):
    client = OpenAI(
        api_key="ollama",  
        base_url="http://localhost:11434/v1"
    )

    completion = client.chat.completions.create(
        model="codellama:34b",  
        messages=[
            {'role': 'user', 
            'content': prompt,}
        ],
    )   

    result = completion.choices[0].message.content
    # usage = completion.usage.total_tokens
    return result

def truncate_code(code, pattern, special_point=None):
    '''
    special_point: for case 'H_M_H_M' and 'M_H_M_H'
    '''
    lines = code.strip('\n').split('\n')
    cut = random.randint(2, len(lines)-1) # 避开第一句def
    cut1 = random.randint(3, len(lines) - 5) if len(lines)-5> 3 else 3
    cut2 = random.randint(cut1 + 2, len(lines) - 1) if len(lines)-1 > cut1 + 2 else cut1 + 2

    begin_code, between_code, end_code= '','',''
    pos = []
    if pattern == 'H_M':
        if(special_point is not None):
            if special_point +1>= len(lines):
                cut = len(lines)  # 或者 continue 跳过
            else:
                cut = random.randint(special_point+1,len(lines))
        begin_code = '\n'.join(lines[:cut])
        pos.append(cut)
    elif pattern == 'M_H':
        if(special_point is not None):
            cut = random.randint(1,special_point) 
        end_code =  '\n'.join(code[cut:])
        pos.append(cut)
    elif pattern == 'H_M_H':
        begin_code = '\n'.join(lines[:cut1])
        end_code = '\n'.join(lines[cut2:])
        pos.append(cut1)
        pos.append(cut2)
    elif pattern == 'M_H_M':
        between_code = '\n'.join(lines[cut1:cut2]) 
        pos.append(cut1)
        pos.append(cut2)      
    
    return begin_code, between_code, end_code, pos

def clean_docstring(code):
    # 匹配def开头到第一个三引号docstring结束
    pattern = r"(^\s*def[^\n]*:\s*\n?\s*)([\"']{3}[\s\S]*?[\"']{3}\n?)"
    match = re.match(pattern, code)
    if match:
        # 保留函数定义行，去除docstring
        return match.group(1) + code[match.end():]
    else:
        # 没有docstring则返回原始代码
        return code

def match_code(full_code, matched_code):
    '''
    return pos: position of matched code
    '''
    def clean(s):
        return ''.join(line.strip() for line in s.split('\n') if line.strip())
    
    if clean(full_code) == clean(matched_code):
        return None
    
    full_lines = clean_docstring(clean(full_code))
    matched_lines = clean(matched_code)
    n, m = len(full_lines), len(matched_lines)
    
    for i in range(n - m + 1):
        if full_lines[i:i + m] == matched_lines:
            return i
    print(f'full_code:\n {full_code}')
    print(f'matched_code:\n {matched_code}')

    return None

def add_statement_label(code, label):
    labeled_statement = []
    for line in code.split('\n'):
        if line.strip():
            labeled_statement.append((line, label))
    return labeled_statement

def generate_data(code_data,source='CodeSearchNet',lang='python',set_ix='train',
                  save_path='./CodeSearchNet/python/train.jsonl'):
    '''
    code_data: list of dict
    '''
    final_dataset = []  
    prompts = ['H_M', 'M_H', 'H_M_H', 'M_H_M', 'H_M_H_M','M_H_M_H']   

    with open(save_path, 'w') as f:
        for idx, code in enumerate(code_data):
            labeled_statment = []
            human_part, machine_part = '', ''

            cleaned_code = clean_docstring(code['code'])  # 清理代码，去除docstring
            prompt_content = code['docstring'] + '\n' + code['code'].strip().split('\n')[0]
            begin_code_2,end_code_2 = None, None

            # 均分6份prompt pattern
            # split_idx = idx % len(prompts)
            
            # 测试prompt是否正确 -----------------------------
            split_idx = 1
            # -----------------------------------------------

            if split_idx in [0, 1, 2, 3]:
                # 构造prompt --------------------------------------
                begin_code, between_code, end_code, pos = truncate_code(cleaned_code, prompts[split_idx])
                
                prompt_patterns = {
                'H_M': f'''
                    You are a senior {lang} developer. Based on the “Function Description” below 
                    and the “Beginning Code” snippet, continuation the function.

                    **Requirements:**  
                    1. Language: Python  
                    2. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    3. The beginning part of your answer should be exactly as same as the "Beginning Code" snippet. 
                    4. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting. 
                    5. Only output the code without any additional text.

                    ### Function Description:
                    {prompt_content}

                    ### Beginning Code:
                    {begin_code}
                    ''', 
                'M_H': f'''
                    You are a senior {lang} developer. Based on the "Function Description" below
                    and the “Ending Code” snippet, generate the required code. 

                    **Requirements:**
                    1. Language: {lang}  
                    2. Write code to finish the function, and use the "Ending Code" as the final section.
                    3. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    4. The final part of your answer should be exactly as same as the "Ending Code" snippet.
                    5. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                    6. Only output the code without any additional text. 
                    
                    ###Function description: 
                    {prompt_content}
                    
                    ###Ending code: 
                    {end_code}
                    ''', 
                'H_M_H': f'''
                    You are a senior {lang} developer. Based on the "Function Description" below, 
                    generate the {lang} program by filling in the code between the "Beginning Code" 
                    and "Ending Code".
 
                    **Requirements:**
                    1. Language: {lang}  
                    2. Use the "Beginning Code" as the start and the "Ending Code" as the end.
                    3. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    4. The beginning part of your answer should be exactly as same as the "Beginning Code" snippet.
                    5. The final part of your answer should be exactly as same as the "Ending Code" snippet.
                    6. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                    7. Only output the code without any additional text. 
                    
                    ###Function description: 
                    {prompt_content}
                     
                    ###Beginning code: 
                    {begin_code}
                     
                    ###Ending code: 
                    {end_code}
                    ''', 
                'M_H_M': f'''
                    You are a senior {lang} developer. Based on the "Function Description" below, 
                    generate the {lang} program, ensuring to include the "in-between code" within 
                    your generated answer.

                    **Requirements:**
                    1. Language: {lang}  
                    2. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    3. Your answer should include the "in-between code" exactly as provided.
                    4. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                    5. Only output the code without any additional text.  
      
                    ###Function description: 
                    {prompt_content}
                    
                    ###In-between code: 
                    {between_code} 
                    ''', 
                }

                final_prompt = prompt_patterns[prompts[split_idx]]       
                
                if split_idx == 0:
                    for attempt in range(5):
                        hybrid_code = get_code_completion(final_prompt)
                        human_part = begin_code
                        machine_part = hybrid_code[len(begin_code):]
                        if match_code(hybrid_code, begin_code) is not None:
                            labeled_statment.extend(add_statement_label(human_part, 'human'))
                            labeled_statment.extend(add_statement_label(machine_part, 'machine'))
                            break  # 成功，跳出尝试循环
                        else:
                            print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx}, attempt {attempt+1}/5.")
                    else:
                        # 5次都失败，跳过本样本
                        continue
                elif split_idx == 1:
                    for attempt in range(5):
                        hybrid_code = get_code_completion(final_prompt)
                        human_part = end_code
                        machine_part = hybrid_code[: -len(end_code)]
                        if match_code(hybrid_code, end_code) is not None:
                            labeled_statment.extend(add_statement_label(human_part, 'human'))
                            labeled_statment.extend(add_statement_label(machine_part, 'machine'))
                            break  # 成功，跳出尝试循环
                        else:
                            print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx}, attempt {attempt+1}/5.")
                    else:
                        # 5次都失败，跳过本样本
                        continue

                elif split_idx == 2:
                    for attempt in range(5):
                        hybrid_code = get_code_completion(final_prompt)
                        human_part = begin_code + end_code
                        machine_part = hybrid_code[len(begin_code):len(hybrid_code) - len(end_code)]
                        if (match_code(hybrid_code, end_code) is not None) and (match_code(hybrid_code, begin_code) is not None):
                            labeled_statment.extend(add_statement_label(begin_code, 'human'))
                            labeled_statment.extend(add_statement_label(machine_part, 'machine'))
                            labeled_statment.extend(add_statement_label(end_code, 'human'))
                            
                            break  # 成功，跳出尝试循环
                        else:
                            print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx}, attempt {attempt+1}/5.")
                    else:
                        # 5次都失败，跳过本样本
                        continue
                
                elif split_idx == 3:
                    for attempt in range(5):
                        hybrid_code = get_code_completion(final_prompt)
                        human_part = between_code
                        matched_pos = match_code(hybrid_code, between_code)
                        if matched_pos is not None:
                            machine_part = hybrid_code[:matched_pos] + hybrid_code[matched_pos + len(between_code):]
                            labeled_statment.extend(add_statement_label(hybrid_code[:matched_pos], 'machine'))
                            labeled_statment.extend(add_statement_label(between_code, 'human'))
                            labeled_statment.extend(add_statement_label(hybrid_code[matched_pos + len(between_code):], 'machine'))
                            break  # 成功，跳出尝试循环
                        else:
                            print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx}, attempt {attempt+1}/5.")
                    else:
                        # 5次都失败，跳过本样本
                        continue

            if split_idx == 4:
                begin_code, _, end_code,pos_1 = truncate_code(cleaned_code, prompts[2])
                prompt_1 = f'''
                    You are a senior {lang} developer. Based on the "Function Description" below, 
                    generate the {lang} program by filling in the code between the "Beginning Code" 
                    and "Ending Code".
 
                    **Requirements:**
                    1. Language: {lang}  
                    2. Use the "Beginning Code" as the start and the "Ending Code" as the end.
                    3. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    4. The beginning part of your answer should be exactly as same as the "Beginning Code" snippet.
                    5. The final part of your answer should be exactly as same as the "Ending Code" snippet.
                    6. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                    7. Only output the code without any additional text. 
                    
                    ###Function description: 
                    {prompt_content}
                     
                    ###Beginning code: 
                    {begin_code}
                     
                    ###Ending code: 
                    {end_code}
                    '''
                
                for attempt in range(5):
                    result_1 = get_code_completion(prompt_1)
                    matched_special = match_code(result_1, end_code)

                    if matched_special is not None:
                        begin_code_2, _, _,pos_2= truncate_code(result_1, prompts[0],special_point=matched_special)
                        pos = pos_1 + pos_2

                        prompt_2 = f'''
                        You are a senior {lang} developer. Based on the “Function Description” below 
                        and the “Beginning Code” snippet, continuation the function.

                        **Requirements:**  
                        1. Language: Python  
                        2. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                        3. The beginning part of your answer should be exactly as same as the "Beginning Code" snippet. 
                        4. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting. 
                        5. Only output the code without any additional text.

                        ### Function Description:
                        {prompt_content}

                        ### Beginning Code:
                        {begin_code_2}
                        '''
                        final_prompt = prompt_1 + '\n\n' + prompt_2
                        
                        for a in range(5):       
                            hybrid_code = get_code_completion(prompt_2)
                            human_part = begin_code + begin_code_2

                            if (match_code(hybrid_code, begin_code) is not None) and (match_code(hybrid_code, begin_code_2) is not None):
                                machine_part = hybrid_code[match_code(hybrid_code,begin_code)+len(begin_code):match_code(hybrid_code, begin_code_2)] + hybrid_code[match_code(hybrid_code, begin_code_2)+len(begin_code_2):]
                                labeled_statment.extend(add_statement_label(begin_code, 'human'))
                                labeled_statment.extend(add_statement_label(hybrid_code[match_code(hybrid_code,begin_code)+len(begin_code):match_code(hybrid_code, begin_code_2)], 'machine'))
                                labeled_statment.extend(add_statement_label(begin_code_2, 'human'))
                                labeled_statment.extend(add_statement_label(hybrid_code[match_code(hybrid_code, begin_code_2)+len(begin_code_2):], 'machine'))
                                break
                            else:
                                print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx} in step 2, attempt {a+1}/5.")
                        
                        else:
                            # 5次都失败，跳过本样本
                            continue
                        
                        break

                    else:
                        print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx} in step 1, attempt {attempt+1}/5.")
                else:
                    # 5次都失败，跳过本样本
                    continue


            if split_idx == 5:
                begin_code, _, end_code,pos_1 = truncate_code(cleaned_code, prompts[2])
                prompt_1 =  f'''
                    You are a senior {lang} developer. Based on the "Function Description" below, 
                    generate the {lang} program by filling in the code between the "Beginning Code" 
                    and "Ending Code".
 
                    **Requirements:**
                    1. Language: {lang}  
                    2. Use the "Beginning Code" as the start and the "Ending Code" as the end.
                    3. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    4. The beginning part of your answer should be exactly as same as the "Beginning Code" snippet.
                    5. The final part of your answer should be exactly as same as the "Ending Code" snippet.
                    6. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                    7. Only output the code without any additional text. 
                    
                    ###Function description: 
                    {prompt_content}
                     
                    ###Beginning code: 
                    {begin_code}
                     
                    ###Ending code: 
                    {end_code}
                    '''
                
                for attempt in range(5):
                    result_1 = get_code_completion(prompt_1)
                    if (match_code(result_1, begin_code) is not None):
                        if(match_code(result_1, begin_code) >1):
                            _, _, end_code_2, pos_2= truncate_code(result_1, prompts[1],special_point=match_code(result_1, begin_code))
                            pos = pos_1 + pos_2
                            prompt_2 = f'''
                                You are a senior {lang} developer. Based on the "Function Description" below
                                and the “Ending Code” snippet, generate the required code. 

                                **Requirements:**
                                1. Language: {lang}  
                                2. Use the "Ending Code" as the final section.
                                3. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                                4. The final part of your answer should be exactly as same as the "Ending Code" snippet.
                                5. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                                6. Only output the code without any additional text. 
                                
                                ###Function description: 
                                {prompt_content}
                                
                                ###Ending code: 
                                {end_code_2}
                                '''
                            final_prompt = prompt_1 + '\n\n' + prompt_2

                            for a in range(5):
                                hybrid_code = get_code_completion(prompt_2)
                                if (match_code(hybrid_code, end_code) is not None) and (match_code(hybrid_code, end_code_2) is not None):
                                    human_part = end_code + end_code_2

                                    machine_part = hybrid_code[:match_code(hybrid_code, end_code_2)] + hybrid_code[match_code(hybrid_code, end_code_2)+len(end_code_2):match_code(hybrid_code, end_code)]
                                    labeled_statment.extend(add_statement_label(hybrid_code[:match_code(hybrid_code, end_code_2)], 'machine'))
                                    labeled_statment.extend(add_statement_label(end_code, 'human'))
                                    labeled_statment.extend(add_statement_label(hybrid_code[match_code(hybrid_code, end_code_2)+len(end_code_2):match_code(hybrid_code, end_code)], 'machine'))
                                    labeled_statment.extend(add_statement_label(end_code_2, 'human'))
                                else:
                                    print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx} in step 2, attempt {a+1}/5.")
                            else:
                                # 5次都失败，跳过本样本
                                continue
                            
                            break
                    else:
                        print(f"Warning: {prompts[split_idx]}: Could not match code for code_id {idx} in step 1, attempt {attempt+1}/5.")
                else:
                    # 5次都失败，跳过本样本
                    continue

            record = {
                "code_id": idx,
                "code_source": source,
                "code_source_id": code['url'],
                "prompt_pattern": prompts[split_idx],
                "prompt_content": final_prompt,
                "original_code": code['code'],
                "hybrid_code": hybrid_code,
                "language": lang,
                "set_ix": set_ix,
                # "labeled_statement": labeled_statment,
                "boundary_ix": pos,
                "boundary_num": len(pos),
                "human_part": human_part,
                "machine_part": machine_part
            }
            final_dataset.append(record)
            print(f'current code number: {idx}')

            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    
    # Save to JSONL files
    # with open(f'{path}/final_train_data.jsonl', 'w') as f:
    #     for item in final_train_data:
    #         f.write(json.dumps(item) + '\n')
    
    # with open(f'{path}/final_test_data.jsonl', 'w') as f:
    #     for item in final_test_data:
    #         f.write(json.dumps(item) + '\n')
        
    # with open(f'{path}/final_valid_data.jsonl', 'w') as f:
    #     for item in final_valid_data:
    #         f.write(json.dumps(item) + '\n')

    
    return final_dataset

def MHMH_step2(allcode,save_path,lang='python'):
    with open(save_path, 'w') as f:
        for idx, code in allcode.iterrows():
            pattern = r"###Function description:\s*(.*?)(?=\s*###Beginning code:)"
            match = re.search(pattern, code['prompt_content'], re.DOTALL)
            prompt_content = match.group(1)

            if (code['machine_part'] is None) or (code['machine_part'] == ''):
                continue
            special_point = match_code(code['hybrid_code'], code['machine_part'])
            print(f'special_point: {special_point}')


            if special_point > 1:
                _, _, end_code_2, pos_2= truncate_code(code['hybrid_code'], 'M_H',special_point=special_point)
                
                prompt_2 = f'''
                    You are a senior {lang} developer. Based on the "Function Description" below
                    and the “Ending Code” snippet, generate the required code. 

                    **Requirements:**
                    1. Language: {lang}  
                    2. Use the "Ending Code" as the final section.
                    3. Output the complete, runnable function. This means including the already‐given lines and your additional code, all in one contiguous block. 
                    4. The final part of your answer should be exactly as same as the "Ending Code" snippet.
                    5. Wrap your entire response in a fenced code block (` ```{lang} ... ``` `) to ensure proper formatting.
                    6. Only output the code without any additional text. 
                    
                    ###Function description: 
                    {prompt_content}
                    
                    ###Ending code: 
                    {end_code_2}
                    '''
                final_prompt = code['prompt_content'] + '\n\n' + prompt_2

                for attempt in range(5):
                    hybrid_code = get_code_completion(prompt_2)

                    boundary1 = match_code(hybrid_code, end_code_2)
                    boundary2 = match_code(hybrid_code, code['machine_part'])

                    if (boundary1 is not None): 
                        if(boundary2 is not None):
                            boundary3 = match_code(hybrid_code, boundary2 + len(code['machine_part']))
                            pos = [boundary1, boundary2, boundary3]      
                            break  # 成功，跳出尝试循环
                        else: 
                            print(f"Warning: Boundary2: Could not match code for code_id {code['code_id']}, attempt {attempt+1}/5.")
                            print(f"hybrid_code:\n{hybrid_code}")
                            print(f"end_code_2:\n{end_code_2}")
                            print(f"machine_part:\n{code['machine_part']}")
                    else:
                        print(f"Warning: Boundary1: Could not match code for code_id {code['code_id']}, attempt {attempt+1}/5.")
                else:
                    # 5次都失败，跳过本样本
                    continue

                record = {
                    "code_id": code['code_id'],
                    "code_source": code['code_source'],
                    "code_source_id": code['code_source_id'],
                    "prompt_pattern": 'M_H_M_H',
                    "prompt_content": final_prompt,
                    "original_code": code['original_code'],
                    "hybrid_code": hybrid_code,
                    "language": lang,
                    "set_ix": code['set_ix'],   
                    # "labeled_statement": labeled_statment,
                    "boundary_ix": pos,
                    "boundary_num": len(pos),
                    # "human_part": human_part,
                    # "machine_part": machine_part
                }

                print(f'current code number: {idx}')

                f.write(json.dumps(record, ensure_ascii=False) + '\n')


# GENERATE DATA -----------------------------------------------------------------
# load data as dataframe
train_data = load_data('CodeSearchNet/python/python/final/jsonl/train')
test_data = load_data('CodeSearchNet/python/python/final/jsonl/test')
valid_data = load_data('CodeSearchNet/python/python/final/jsonl/valid')

print(len(train_data), len(test_data), len(valid_data)) # 412178 22176 23107

# extract code with len(code_tokens)>200
all_data = train_data + test_data + valid_data
code_data = extract_code(all_data,min_length=200)

# print(type(code_data)) # <class 'list'>
# print(type(code_data[0])) # <class 'dict'>
# print(code_data[0].keys()) # dict_keys(['code_id', 'code_source', 'code_source_id', 'prompt_pattern', 'prompt_content', 'original_code', 'hybrid_code', 'language', 'set_ix', 'labeled_statement', 'boundary_ix', 'boundary_num', 'human_part', 'machine_part'])

# Split data into train/valid/test sets (80:10:10)
total_size = len(code_data)
train_size = int(total_size * 0.8)
test_size = int(total_size * 0.1)

final_train_data = code_data[:train_size]
final_test_data = code_data[train_size:train_size + test_size]
final_valid_data = code_data[train_size + test_size:]

print(f"Total data size: {total_size}") #60904
print(f"Training data size (80%): {len(final_train_data)}") # 48723
print(f"Validation data size (10%): {len(final_valid_data)}") #6091
print(f"Test data size (10%): {len(final_test_data)}") #6090


# sampled_data = random.sample(final_train_data, 1000)
final_train_data = generate_data(final_train_data[20000:25000], source='CodeSearchNet', 
                                 lang='python', set_ix='train',save_path='./CodeSearchNet/python/MH_20000-25000.jsonl')
# --------------------------------------------------------------------------------
#pprint(final_train_data[0]) # dict_keys(['code_id', 'code_source', 'code_source_id', 'prompt_pattern', 'prompt_content', 'original_code', 'hybrid_code', 'language', 'set_ix', 'labeled_statement', 'boundary_ix', 'boundary_num', 'human_part', 'machine_part'])


# # Read as text first
# with open('./CodeSearchNet/python/HMH_3500-5000.jsonl', 'r') as f:
#     lines = f.readlines()

# # Try parsing each line individually
# data = []
# for line in lines:
#     data.append(json.loads(line))

# df = pd.DataFrame(data)
# MHMH_step2(df, save_path='CodeSearchNet/python/MHMH_3500-5000.jsonl', lang='python')
# #pprint(df.to_dict(orient='records'))  # Print first 5 records