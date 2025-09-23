# Lenet on Sambanova

##### Go to direcotry with Lenet

```bash
cd ~/apps/starters/lenet
```

##### Activate the Virtual Environemnt 

```bash
source /opt/sambaflow/apps/starters/lenet/venv/bin/activate
```

##### Compile and train the LeNet model

```bash
srun python lenet.py compile -b=1 --pef-name="lenet" --output-folder="pef"
srun python lenet.py run --pef="pef/lenet/lenet.pef"
```
