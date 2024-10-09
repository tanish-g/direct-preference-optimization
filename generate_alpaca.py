import json
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import time
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
import torch
from tqdm.auto import tqdm
import json
from datasets import load_dataset
import csv
import argparse
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import pickle
import pdb
import datasets
 
 
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to(accelerator.device) for stop in stops]
 
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False
 
 
if __name__=='__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None )
    parser.add_argument('--model_path', type=str, default='model_path', help='Model name')
    args = parser.parse_args()


    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
   
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
   
    model_name = args.model_name
   
    data = []
    for i in range(len(eval_set)):
        item = {}
        item['instruction'] = eval_set[i]['instruction']
        item['id'] = i
        data.append(item)
 
    print(model_name.split('/')[-1])
   
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = {"": accelerator.process_index}, trust_remote_code=True, torch_dtype=torch.bfloat16,cache_dir = '/tmp/models/')
   
    stop_words = ["</s>"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
 
   
    accelerator.wait_for_everyone()
    start = time.process_time()
   
    prompts_all = data
   
    with accelerator.split_between_processes(prompts_all) as prompts:
        results=dict(outputs=[])
   
        for item in tqdm(prompts):
            
            prompt = [{"role": "system", "content": "You are a helpful AI assistant."},{"role": "user", "content": item['instruction']}]
            prompt_with_template = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,return_tensors='pt').to(accelerator.device)
            output = model.generate(input_ids = prompt_with_template, max_new_tokens= 512, stopping_criteria=stopping_criteria)
            output_text = tokenizer.decode(output[:, prompt_with_template.shape[-1]:][0], skip_special_tokens=True)
            output_item = {
                "instruction": item['instruction']
                "output": output_text
                "generator": model_path.split('/')[-1]
            }
            results['outputs'].append(output_item)
   
        results = [results]
   
    results_gathered=gather_object(results)
   
    if accelerator.is_main_process:
        final_results = []
        for i, result in enumerate(results_gathered):
            final_results.extend(result['outputs'])
        print(len(final_results))
        print("Time to process examples:-", time.process_time()-start)
       
        with open(f"/mnt/azureml/cr/j/694fe47564604d2bb5fa17e72155dbb1/exe/wd/outputs/alpaca_evaluation_json/alpaca_results_{model_name.split('/')[-1]}.json", 'w') as f:
                json.dump(final_results, f, indent=4)
