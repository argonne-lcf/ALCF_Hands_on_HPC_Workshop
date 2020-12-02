# Building a CPU-side data pipeline

New AI systems largely depend on CPU-GPU hybrid architectures. This makes efficient use of CPU-side resources important in order to feed sufficient data to the GPU algorithms. Ideally, the CPU processes data and builds training batches, while the GPU performs the compute intensive forward and backward gradient calculations.

Here there are examples of building a data pipeline for both Tensorflow and PyTorch. Tensorflow's data pipeline API is a bit more advanced than PyTorch so we'll focus on that one first.

# ImageNet Dataset

This example uses the ImageNet dataset to build training batches.

![Turtle](images/n01667778_12001.JPEG) ![Dog](images/n02094114_1205.JPEG)



