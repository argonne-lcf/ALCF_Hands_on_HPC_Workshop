# GPT 1.5B on Sambanova

## GPT 1.5B

The GPT 1.5B application example is provided in the the path : `/opt/sambaflow/apps/nlp/transformers_on_rdu/`.
The scripts containing the `compile` and `run` commands for the GPT 1.5B model can be accessed at the path `/data/ANL/scripts/1.24.1/legacy_models/Gpt1.5B_base_single_compile.sh` and `/data/ANL/scripts/1.24.1/legacy_models/Gpt1.5B_base_single_run.sh` on any SN30 compute node. This script is compiled and run for only 1 instance and the model fits on 4 tiles or half of a RDU. The scripts are provided for reference. 

Change directory and copy files.

```bash
mkdir -p ~/apps/nlp/Gpt1.5B_single
cd ~/apps/nlp/Gpt1.5B_single
```

Copy and paste the contents of
[Gpt1.5B_base_single_compile.sh](./files/Gpt1.5B_base_single_compile.sh "Gpt1.5B_base_single_compile.sh") and [Gpt1.5B_base_single_run.sh](./files/Gpt1.5B_base_single_run.sh "Gpt1.5B_base_single_run.sh") 
to a file with the same names into the current directory using your favorite editor.

or copy the contents from `/data/ANL/scripts/Gpt1.5B_base_single_compile.sh` and `/data/ANL/scripts/Gpt1.5B_base_single_run.sh`.

```bash
cp /data/ANL/scripts/1.24.1/legacy_models/Gpt1.5B_base_single_compile.sh ~/apps/nlp/Gpt1.5B_single/
cp /data/ANL/scripts/1.24.1/legacy_models/Gpt1.5B_base_single_run.sh ~/apps/nlp/Gpt1.5B_single/
```
If you have already compiled for a previous version of the sambaflow stack, delete existing pef file, if it exists.
```bash
rm /data/scratch/$(whoami)/GPT1.5B_base_single_32/GPT1.5B_base_single_32/GPT1.5B_base_single_32.pef
```

Run the script with batch size as an argument(shown below with an example of 32).

```bash
chmod +x Gpt1.5B_base_single_compile.sh 
./Gpt1.5B_base_single_compile.sh 32
```

The Gpt1.5B_base_single_compile.sh  script will internally call the Gpt1.5B_base_single_run.sh to perform the training. You can inspect the `compile` and `run` commands in the scripts to learn that this model trains with a batch size of 32 for 1 instance over 4 tiles. The human decision file and the compiler config file helps to optimize the compute and memory resources specific to this Gpt 1.5B model run.

```bash
python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py compile --pef-name=GPT1.5B_base_single_32 --output-folder=/data/scratch/user/GPT1.5B_base_single_32 --module_name gpt2_pretrain --task_name clm --max_seq_length 1024 -b 32  --output_dir=/data/scratch/user/GPT1.5B_base_single_32/hf_gpt1dot5b_ss1k_gas_1_bs32  --overwrite_output_dir --do_train  --per_device_train_batch_size 32   --tokenizer_name gpt2 --model_name gpt2 --mac-v2 --non_split_head --mac-human-decision /opt/sambaflow/apps/nlp/transformers_on_rdu/human_decisions_gm/mac_v2_overrides/gpt2_48_enc_full_recompute_training_spatialmapping_tiling16_clmerge_gm_pardp2_lnsd.json --compiler-configs-file /opt/sambaflow/apps/nlp/transformers_on_rdu/human_decisions_gm/compiler_configs/compiler_configs_gpt1dot5b_perf.json --skip_broadcast_patch --config_name /opt/sambaflow/apps/nlp/transformers_on_rdu/customer_specific/mv/configs/gpt2_config_xl_50260.json --no_index_select_patch --weight_decay 0.1  --max_grad_norm_clip 1.0 --num-tiles 4 --enable-stochastic-rounding
```

```bash
COMMAND= /usr/local/bin/srun --mpi=pmi2 python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py run  -b 32  --data_dir /data/ANL/ss1024 --pef=/data/scratch/user/GPT1.5B_base_single_32/GPT1.5B_base_single_32/GPT1.5B_base_single_32.pef --output_dir=/data/scratch/user/GPT1.5B_base_single_32/hf_gpt1dot5b_ss1k_gas_1_bs16 --module_name gpt2_pretrain --task_name clm --max_seq_length 1024  --overwrite_output_dir --do_train  --per_device_train_batch_size 32 --tokenizer_name gpt2 --model_name gpt2 --non_split_head --skip_broadcast_patch --no_index_select_patch --config_name /opt/sambaflow/apps/nlp/transformers_on_rdu/customer_specific/mv/configs/gpt2_config_xl_50260.json --max_grad_norm_clip 1.0 --skip_checkpoint --logging_steps 1 --max_steps 75000 --learning_rate 0.00025 --steps_this_run 100
```

The `sntilestat` command shows that the application runs on 4 tiles as shown below.

```bash
/XRDU_0/RDU_0/TILE_0   2.1  96.9    0.8    0.1    0.0      0.0 796481  user python /opt/sambaflow/apps/nlp/transformers_on_rdu/
/XRDU_0/RDU_0/TILE_1   2.1  96.9    0.8    0.1    0.0      0.0 796481  user python /opt/sambaflow/apps/nlp/transformers_on_rdu/
/XRDU_0/RDU_0/TILE_2   2.5  96.9    0.4    0.1    0.0      0.0 796481  user python /opt/sambaflow/apps/nlp/transformers_on_rdu/
/XRDU_0/RDU_0/TILE_3   2.5  96.9    0.4    0.1    0.0      0.0 796481  user python /opt/sambaflow/apps/nlp/transformers_on_rdu/
/XRDU_0/RDU_0/TILE_4 100.0   0.0    0.0    0.0    0.0      0.0
/XRDU_0/RDU_0/TILE_5 100.0   0.0    0.0    0.0    0.0      0.0
...

```

