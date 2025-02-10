import re
from config import DatasetEnum, DatasetConfig

def gsm8k_extract_final_answer(response: str) -> str | None:
    """
    :param pred: generated text
    :param dataset: task
    :return: [cleaned_text, final_prediction]
    """

    WRONG_ANSWER = None

    # pred = pred.strip().split('\n\n')[0]
    pred = response.strip().lower()
    temp = pred
    temp_list = [temp]

    struct_ans_flag = False
    for answer_prefix in ['####', '###', 'answer:', '\nAnswer', 'answer is']:
        if answer_prefix in pred:
            temp_list = pred.split(answer_prefix)[1:]
            struct_ans_flag = True
            break
    
    for temp in temp_list:
        temp_ori = [item for item in re.findall(r'-?\d+\.?\$?,?\d*', temp)]
        # temp = [item.strip('.') for item in re.findall(r'-?\d+\.?\d*', temp.replace(',', ''))]
        temp = [str(int(float(item.strip('.'))) if float(item.strip('.')).is_integer() else float(item.strip('.'))) for item in re.findall(r'-?\d+\.?\d*', temp.replace(',', ''))]
        if len(temp) > 0: break
    
    if len(temp) == 0:
        final_pred = WRONG_ANSWER
        if struct_ans_flag:
            answer_prefix_idx = pred.index(answer_prefix)
            next_word = pred[answer_prefix_idx+len(answer_prefix):].split()
            if len(next_word) == 0:
                next_word = ''
            elif next_word[0] == ':':
                if len(next_word) == 1:
                    next_word = ' '
                else:
                    next_word = ': ' + next_word[1]
            else:
                next_word = ' ' + next_word[0]
            pred = pred[:answer_prefix_idx+len(answer_prefix)] + next_word

    elif struct_ans_flag:
        final_pred = temp[0]
        answer_prefix_idx = pred.index(answer_prefix)
        if final_pred in pred[answer_prefix_idx:]:
            temp_idx = pred[answer_prefix_idx:].index(final_pred)
            pred = pred[:answer_prefix_idx + temp_idx + len(final_pred)]
        else:
            next_word = pred[answer_prefix_idx+len(answer_prefix):].split()
            if next_word[0] == ':':
                next_word = ': ' + next_word[1]
            else:
                next_word = ' ' + next_word[0]
            pred = pred[:answer_prefix_idx + len(answer_prefix)] + next_word
            
    elif not struct_ans_flag:
        final_pred = temp[-1]
        if final_pred in pred:
            pred = pred[:pred.index(final_pred) + len(final_pred)]
        elif temp_ori[-1] in pred:
            pred = pred[:pred.index(temp_ori[-1]) + len(temp_ori[-1])]
        else:
            pass
    else:
        raise RuntimeError()

    return final_pred

def extract_qa_final_answer(response: str) -> str | None:

    response = response.lower().strip()

    # Pattern to match the final answer (e.g., Answer: A, Answer: B, Answer: A. etc.)
    match = re.search(r'\bAnswer:\s*([A-D])\.?\b', response, re.IGNORECASE)
    
    if match:
        # Return the answer (A, B, C, D) in uppercase if 'Answer: X' is found
        return match.group(1).upper()
    else:
        # If no 'Answer: X' is found, search for the last occurrence of A, B, C, or D
        last_option_match = re.findall(r'\b([A-D])\b', response, re.IGNORECASE)
        
        if last_option_match:
            # Return the last mentioned answer (in uppercase)
            return last_option_match[-1].upper()
        else:
            # No valid answer found
            return None

def extract_final_answer(response: str, dataset: DatasetEnum) -> str:
    if dataset in DatasetConfig().get_math_tasks():
        func = gsm8k_extract_final_answer
    elif dataset in DatasetConfig().get_qa_tasks():
        func = extract_qa_final_answer
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    final = func(response)
    if final is None: final = "None"

    return final.lower()


if __name__ == '__main__':
    from config import InitConfig, parse_args, DatasetEnum
    from dataclasses import dataclass
    from tools import tools_json_load, tools_json_dump

    @dataclass
    class Args(InitConfig):
        dataset: DatasetEnum = DatasetEnum.gsm8k_test
        file: str | None = None
    
    args: Args = parse_args(Args, pass_in=[])

    uuids = {item['uuid'] for item in tools_json_load(args.dataset.value)}
    data: dict = tools_json_load(args.file)

    total = []
    for uuid, item in data.items():
        if uuid in uuids:
            total.append(item)
    
    ori_acc = sum([item['correct'] for item in total]) / len(total) * 100
    now_acc = sum([extract_final_answer(item['responses'], args.dataset) == str(item['label']).lower() for item in total]) / len(total) * 100

    print(f"total={len(total)}, ori acc={ori_acc:.3f}, now acc={now_acc:.3f}")

    wrong_examples = [item for item in total if extract_final_answer(item['responses'], args.dataset) != str(item['label']).lower()]
    tools_json_dump(wrong_examples, '/tmp/test.json')






