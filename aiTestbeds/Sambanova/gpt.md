# GPT on Sambanova

##### Create and and move to the following directory.

```bash
mkdir ~/apps/nlp/Gpt1.5B_single
cd ~/apps/nlp/Gpt1.5B_single
```

##### Copy script to Compile and Run

```bash
cp /data/ANL/scripts/Gpt1.5B_base_compile.sh .
cp /data/ANL/scripts/Gpt1.5B_base_run.sh .

chmod +x Gpt1.5B_base_compile.sh
chmod +x Gpt1.5B_base_run.sh
```

##### Run the script to Compile and Run

```bash
./Gpt1.5B_base_compile.sh 32
./Gpt1.5B_single.sh 32
```
