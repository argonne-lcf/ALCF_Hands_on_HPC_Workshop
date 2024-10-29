# Sambanova

## Connection to Sambanova 

![Sambanova connection diagram](./sambanova_login.jpg)

Login to the Sambanova login node from your local machine.
Once you are on the login node, ssh to one of the sambanova nodes.

```bash
local > ssh ALCFUserID@sambanova.alcf.anl.gov
sm-login-1 > ssh sn30-r1-h1       
```

You can also ssh to `sn30-r1-h1 , sn30-r1-h2, sn30-r2-h1, sn30-r2-h2, sn30-r3-h1, sn30-r3-h2, sn30-r4-h1, sn30-r4-h2`

## Sambanova Examples

We use examples from Sambanova for this hands-on. 
Copy those examples to your home directory. 
```bash
cp -r /opt/sambaflow/apps/ ~
```

## Create Virtual Environment 

Sambanova software stack and associated environmental variables are setup at login. 

Each of the samples or application examples provided by SambaNova has its own pre-built virtual environment which can be readily used. They are present in the `/opt/sambaflow/apps/` directory tree within each of the applications. 

## Hands-on Session Example 

* [GPT](./gpt.md)

## Next Steps 

* [Lenet](./lenet.md)

## Useful Directories 

* Sambanova Applications Path : `/opt/sambaflow/apps`
* Sambanova Model Scripts : `/data/ANL/scripts`
* Important Datasets  : `/software/sambanova/dataset`

# Useful Resources 

* [ALCF Sambanova Documentation](https://docs.alcf.anl.gov/ai-testbed/sambanova_gen2/getting-started/)
* [Sambanova Documentation](https://docs.sambanova.ai/developer/latest/sambaflow-intro.html) 
* Sambanova Applications Path: `/opt/sambaflow/apps/`
* Sambanova Model scripts: `/data/ANL/scripts/`
* Important Datasets: `/software/sambanova/dataset/`
