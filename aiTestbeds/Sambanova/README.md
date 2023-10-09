# Sambanova

## Connection to Graphcore 

![Sambanova connection diagram](./sambanova_login.jpg)

Login to the Sambanova login node from your local machine.
Once you are on the login node, ssh to one of the sambanova nodes.

```bash
ssh ALCFUserID@sambanova.alcf.anl.gov

ssh sn30-r1-h1
```
You can also ssh to `sn30-r1-h1 , sn30-r1-h2, sn30-r2-h1, sn30-r2-h2, sn30-r3-h1, sn30-r3-h2, sn30-r4-h1, sn30-r4-h2`


## Create Virtual Environment 

Sambanova software stack and associated environmental variables are setup at login. 


```bash
python -m venv --system-site-packages my_env 
source my_env/bin/activate
```

## Sambanova Examples

We use examples from Sambanova for this hands-on. 
Copy those examples to your home directpry. 
```bash
cp -r /opt/sambaflow/apps/ ~
```

## Run Examples 

* [Lenet](./lenet.md)
* [GPT](./gpt.md)

## Useful Directories 

* Sambanova Applications Path : `/opt/sambaflow/apps`
* Sambanova Model Scripts : `/data/ANL/scripts`
* Important Datasets  : `/software/sambanova/dataset`

# Useful Resources 

* [ALCF Sambanova Documentation](https://docs.alcf.anl.gov/ai-testbed/sambanova_gen2/getting-started/)
* [Sambanova Documentation](https://docs.sambanova.ai/developer/latest/sambaflow-intro.html) 