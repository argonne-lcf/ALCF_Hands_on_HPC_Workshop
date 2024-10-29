# Simple GEMV Example using Simulator


This example demonstrates a complete CSL program using single PE using Cerebras SDK Simulator. 
This directory consists of:

1. [run.py](./run.py) : The host program is defined in this script.
   
2. [layout.csl](./layout.csl) : Layout Code in CSL. 

3. [pe_program.csl](./pe_program.csl) : Device Code in CSL. 

4. [commands.sh](./commands.sh) : Compiles CSL files and runs host code in a single shell script. 


## Run Example

```bash
$ bash commands.sh
```

This scripts  first compiles the CSL code files, and then invokes the host program. 



