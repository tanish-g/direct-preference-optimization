import torch
import transformers
if __name__=='__main__':
  model_name_or_path = "lxuechen/phi-2-sft"
  model= transformers.AutoModelForCausalLM.from_pretrained(
      model_name_or_path,
      low_cpu_mem_usage=True,
      trust_remote_code=True,
      torch_dtype=torch.float16,
  )
   
  state_dict = torch.load("/home/azureuser/cloudfiles/code/Users/t-sshandilya/sallms/dpo_train/direct-preference-optimization/.cache/azureuser/train_phi2_dpo_scores_greater_than_3_org_1e_6_2024-09-26_15-12-27_602430/LATEST/policy.pt", map_location='cpu')
   
  model.load_state_dict(state_dict['state'])
   
  model.save_pretrained("/home/azureuser/cloudfiles/code/Users/t-sshandilya/sallms/dpo_train/direct-preference-optimization/trained_models_hf/phi-2-UF-2-epochs-scores_greater_than_3-baseline-lr_1e_6", safe_serialization=True)
 
 
