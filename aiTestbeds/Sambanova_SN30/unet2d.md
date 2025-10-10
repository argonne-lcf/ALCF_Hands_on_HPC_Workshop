# Unet on Sambanova

## UNet2D

The UNet application example is provided in the the path : `/opt/sambaflow/apps/image/segmentation/`. As any other application, we first compile and then train the model using *compile* and *run* arguments respectively.
The scripts containing the compile and run commands for UNet2D model can be accessed at [Unet2d.sh](./Unet2d.sh "Unet2d.sh") or at `/data/ANL/scripts/Unet2d.sh` on any SN30 compute node.

Change directory and copy files.

```bash
mkdir -p ~/apps/image/unet
cd ~/apps/image/unet
```

Copy and paste the contents of
[Unet2d.sh](./Unet2d.sh "Unet2d.sh")
to a file with the same name into the current directory using your favorite editor.

```bash
chmod +x Unet2d.sh
```

Run these commands for training (compile + train):

```bash
./Unet2d.sh compile <image size> <batch_size> <num of instances> <RunID>
./Unet2d.sh run <image size> <batch_size> <num of instances> <RunID>
```

The `compile` and `run` arguments of the script can only be run with number of instances equal to 1, indicating that this is a simple 4 tile run without data parallel framework.
For a image size of 256x256 and batch size 256 when running just 1 instance, the commands are provided as follows.

!!! note

    The compilation runs for over 30 minutes.

```bash
./Unet2d.sh compile 256 256 1 unet2d_single_compile
./Unet2d.sh run 256 256 1 unet2d_single_run
```

The above commands displays the file that contains the output for the execution of the above scripts, usually `/data/ANL/results/<hostname>/<userid>/<RunID>/Unet2d.out`

If we inspect the compile and run commands for the UNet application provided in the script, we see that the application is compiled with `--num-tiles 4`, which means that the entire application fits on 4 tiles or half of a RDU.
The pef generated from the compilation process of the above command is placed under `out/Unet2d/unet_train_256_256_single_4` inside the current working directory.

```bash
python ${UNET}/compile.py compile --mac-v2 --in-channels=3 --in-width=${2} --in-height=${2} --batch-size=${BS} --enable-conv-tiling --num-tiles=4 --pef-name=unet_train_${BS}_${2}_single_${NUM_TILES} --output-folder=${OUTDIR}
```

```bash
srun --nodelist $(hostname) python /opt/sambaflow/apps/image/segmentation//hook.py run --data-cache=${CACHE_DIR}  --data-in-memory --num-workers=${NUM_WORKERS} --enable-tiling  --min-throughput 395 --in-channels=3 --in-width=${2} --in-height=${2} --init-features 32 --batch-size=${BS} --epochs 10 --data-dir ${DS} --log-dir log_dir_unet_${2}_${BS}_single_${NUM_TILES} --pef=${OUTDIR}/unet_train_${BS}_${2}_single_${NUM_TILES}/unet_train_${BS}_${2}_single_${NUM_TILES}.pef
```

The performance data is located at the bottom of log file.

```console
inner train loop time : 374.6789753437042 for 10 epochs, number of global steps: 130, e2e samples_per_sec: 88.82270474202953
```
