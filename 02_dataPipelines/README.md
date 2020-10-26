# Building a CPU-side data pipeline

New AI systems largely depend on CPU-GPU hybrid architectures. This makes efficient use of CPU-side resources important in order to feed sufficient data to the GPU algorithms. Ideally, the CPU processes data and builds training batches, while the GPU performs the compute intensive forward and backward gradient calculations.