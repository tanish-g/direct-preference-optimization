
python -u train.py model=phi3_mini datasets=[uf] loss=dpo loss.beta=0.1 lr=1e-6 exp_name=train_phi3_mini_sft_random_21k_dpo_1e_6 gradient_accumulation_steps=1 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 n_epochs=2
python -u train.py model=phi3_mini datasets=[base_uf] loss=dpo loss.beta=0.1 lr=1e-6 exp_name=train_phi3_mini_sft_baseline_dpo_1e_6 gradient_accumulation_steps=1 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 n_epochs=2
python -u train.py model=phi3_mini datasets=[slac_uf] loss=dpo loss.beta=0.1 lr=1e-6 exp_name=train_phi3_mini_sft_ours_hal_dpo_1e_6 gradient_accumulation_steps=1 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 n_epochs=2

