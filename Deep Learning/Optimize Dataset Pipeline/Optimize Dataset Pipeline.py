import tensorflow as tf
import time

class FileDataset(tf.data.Dataset):
    def read_files_in_batches(num_samples):
        time.sleep(0.03)
        for sample_idx in range(num_samples):
            time.sleep(0.015)
            yield (sample_idx,)
    
    def __new__(cls,num_samples=3):
        return tf.data.Dataset.from_generator(
            cls.read_files_in_batches,
            output_signature=tf.TensorSpec(shape=(1,), dtype=tf.int64),
            args=(num_samples,)
        )

def benchmark(dataset, num_epochs=2):
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.01)

start = time.time()
benchmark(FileDataset(),5)
end = time.time()

print("Execution time Normal:", end - start)

start = time.time()
benchmark(FileDataset().prefetch(tf.data.experimental.AUTOTUNE),5)
end = time.time()

print("Execution time Prefetch:", end - start)

start = time.time()
benchmark(FileDataset().cache(),5)
end = time.time()

print("Execution time Cache:", end - start)

