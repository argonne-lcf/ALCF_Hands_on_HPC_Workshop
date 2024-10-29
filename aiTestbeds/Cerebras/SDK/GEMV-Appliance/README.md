# Simple GEMV Example using Appliance Mode


This example demonstrates a complete CSL program using single PE using Cerebras SDK Applicance Mode. 
This directory consists of:

1. [run.py](./run.py) : The host program is defined in this script.

2. [compile.py](./compile.py) : This python script is used to compile CSL files for Appliance mode.
   
3. [layout.csl](./layout.csl) : Layout Code in CSL. 

4. [pe_program.csl](./pe_program.csl) : Device Code in CSL. 

5. [commands.sh](./commands.sh) : Compiles CSL files and runs host code in a single shell script. 


## Run Example

```bash
$ bash commands.sh
```

This scripts first compiles the CSL code files, and then invokes the host program. 



