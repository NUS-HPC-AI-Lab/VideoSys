# Data-Centric Parallel (DCP)

[[blog](https://oahzxl.github.io/DCP/)]

- [Data-Centric Parallel (DCP)](#pyramid-attention-broadcastpab)
  - [Insights](#insights)
  - [Pyramid Attention Broadcast (PAB) Mechanism](#pyramid-attention-broadcast-pab-mechanism)
  - [Experimental Results](#experimental-results)
  - [Usage](#usage)
    - [Supported Models](#supported-models)
    - [Configuration for PAB](#configuration-for-pab)
      - [Parameters](#parameters)
      - [Example Configuration](#example-configuration)


We introduce Data-Centric Parallel (DCP), a simple but effective approach to accelerate distributed training of varaible sequences (e.g., videos). Unlike previous methods that fix training settings, DCP dyanmically adjusts parallelism and other configs driven by incoming data during runtime. This method significantly reduces communication overhead and computational inefficiencies, achieving up to 2.1x speedup. As a ease-of-use method, our method can enpower any models and most parallel methods within minimal code changes (10 lines).

## Insights

![method](../assets/figures/dcp_compare.png)

Variable sequences training is commonly used in various tasks, such as video generation (Sora, Movie Gen, Open-Sora, HunyuanVideo), text generation (Llama-3), and scientific computing (AlphaFold). Such strategy offers them two advantages: enhanced generation quality and flexible output sizes.

We compare two common parallel methods for variable sequences training in Figure 1. Bucket parallel fixes sequence parallel size based on the longest sequence, adjusting batch sizes for balance. Packed parallel, while also fixing sequence parallel size, concats multiple sequences in one batch.

However, bucket parallel struggles with workload imbalance at small batch sizes and communication inefficiency for shorter sequences, while packed parallel, despite better load balancing, incurs unnecessary communication overhead and requires complex implementation changes. Both approaches fail to addresses two critical challenges in variable sequences training: necessary sequence parallel for long sequences and the unbalanced workloads from diverse sequence sizes, due to their fixed settings and lack of data awareness.

## Data-Centric Parallel (DCP) Mechanism

![method](../assets/figures/pab_method.png)

Building on these insights, we propose a **pyramid attention broadcast (PAB)** mechanism to minimize unnecessary computations and optimize the utility of each attention module, as shown in Figure[xx figure] below.

In the middle segment, we broadcast one step's attention outputs to its subsequent several steps, thereby significantly reducing the computational cost on attention modules.

For more efficient broadcast and minimum influence to effect, we set varied broadcast ranges for different attentions based on their stability and differences.
**The smaller the variation in attention, the broader the potential broadcast range.**


## Experimental Results
Here are the results of our experiments, more results are shown in https://oahzxl.github.io/DCP/:

![pab_vis](../assets/figures/dcp_speedup.png)

## Usage

DCP consists of two main stages: data profile and data-centric runtime.
With the same configuration parameters, VideoSys will automatically run the profiling stage or detect the previous profiling results and start the training with the data-centric runtime.

### Supported Models

DCP currently supports Open-Sora.

### Configuration for DCP

To enable the Data-centric Parallel (DCP), configure the following parameters to use different variants of DCP.

#### Parameters
- **dynamic_sp**: Enable or disable variable sequence parallelism for all DCP variants.
  - Type: `True` or `False`

- **dynamic_recompute**: Enable or disable pyramid activation checkpointing (PAC), exclusive to DCP-inter to better exploit GPU memory for shorter sequences.
  - Type: `True` or `False`

- **auto_grad_accumulation**: Enable or disable variable number of gradient accumulation steps for DCP-inter.
  - Type: `True` or `False`

- **max_grad_accumulation_steps**: The maximum number of gradient accumulation steps for DCP-inter.
  - Type: Integer

- **min_grad_accumulation_steps**: The minimum number of gradient accumulation steps for DCP-inter.
  - Type: Integer

- **calculate_imbalance**: Enable or disable logging estimated imbalance rate.
  - Type: `True` or `False`

#### Example Configuration

```yaml
# DCP-intra
dynamic_sp: true

# DCP-inter
dynamic_sp: true
auto_grad_accumulation: true

# DCP-inter+PAC
dynamic_sp: true
dynamic_recompute: true
auto_grad_accumulation: true
```

Explanation:

- **DCP-intra**:
  - Only set variable sequence parllelism to enable DCP-intra (`dynamic_sp: true`)

- **DCP-inter**:
  - Set variable sequence parllelism (`dynamic_sp: true`)
  - Set variable number of gradient accumulation steps to enable DCP-inter (`auto_grad_accumulation: true`)

- **DCP-inter**:
  - Set variable sequence parllelism (`dynamic_sp: true`)
  - Set variable number of gradient accumulation steps (`auto_grad_accumulation: true`)
  - Enable PAC over DCP-inter setup (`auto_grad_accumulation: true`)

Adjust these settings based on your specific needs to optimize the performance.

For complete examples, please refer to [dcp_launch.sh](examples/training/open_sora/dcp_launch.sh)
