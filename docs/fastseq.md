# FastSeq

![fastseq_overview](./figure/fastseq_overview.png)

In the realm of visual generation models, such as DiT, sequence parallelism is indispensable for effective long-sequence training and low-latency inference. Two key features can summarize the distinctive nature of these tasks:

- The model parameter is smaller compared with LLMs, but the sequence can be very long, making communication a bottleneck.
- As the model size is relatively small, it only needs sequence parallelism within a node.

However, existing methods like DeepSpeed-Ulysses and Megatron-LM Sequence Parallelism face limitations when applied to such tasks. They either introduce excessive sequence communication or lack efficiency in handling small-scale sequence parallelism.

To this end, we present FastSeq, a novel sequence parallelism for large sequences and small-scale parallelism. Our method focuses on minimizing sequence communication by employing only two communication operators for every transformer layer. We leverage AllGather to enhance communication efficiency, and we strategically employ an async ring to overlap AllGather communication with qkv computation, further optimizing performance.

Here are the results of our experiments, more results will be coming soon:

![fastseq_exp](./figure/fastseq_exp.png)
