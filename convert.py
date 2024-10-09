import torch
from transformers import AutoModelForCausalLM

if __name__ == '__main__':
    model_name_or_path = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir='/tmp/.cache/azureuser/',
        
    )

    load_paths = [
        '/tmp/.cache/azureuser/train_phi3_mini_sft_random_21k_dpo_1e_6_2024-10-09_06-29-49_857828',
        '/tmp/.cache/azureuser/train_phi3_mini_sft_baseline_dpo_1e_6_2024-10-09_08-24-18_504015',
        '/tmp/.cache/azureuser/train_phi3_mini_sft_ours_hal_dpo_1e_6_2024-10-09_10-16-25_489121'
    ]

    save_paths = [
        '/mnt/azureml/cr/j/694fe47564604d2bb5fa17e72155dbb1/exe/wd/outputs/train_phi3_mini_sft_random_21k_dpo_1e_6',
        '/mnt/azureml/cr/j/694fe47564604d2bb5fa17e72155dbb1/exe/wd/outputs/train_phi3_mini_sft_baseline_dpo_1e_6',
        '/mnt/azureml/cr/j/694fe47564604d2bb5fa17e72155dbb1/exe/wd/outputs/train_phi3_mini_sft_ours_hal_dpo_1e_6'
    ]

    for i in range(3):
        # Load the saved state dict
        state_dict = torch.load(f"{load_paths[i]}/LATEST/policy.pt", map_location='cpu')
        
        # Loading the state dict into the model
        model.load_state_dict(state_dict['state'])
        model.save_pretrained(
            save_paths[i],
            safe_serialization=True,
        )
        # Save the model with safe serialization enabled
        # model.save_pretrained(f'{save_paths[i]}', safe_serialization=True)
