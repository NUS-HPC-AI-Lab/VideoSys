import os
from functools import partial
from typing import Any, Optional, List, Dict, Tuple
from itertools import islice, repeat
from collections import defaultdict

import torch
import torch.distributed as dist
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import videosys

from .mp_utils import ProcessWorkerWrapper, ResultHandler, WorkerMonitor, get_distributed_init_method, get_open_port
from .ray_utils import RayWorkerWrapper, get_ip, initialize_ray_cluster
import logging
logger = logging.getLogger(__name__)

class VideoSysEngine:
    """
    this is partly inspired by vllm
    """

    def __init__(self, config):
        self.config = config
        self.parallel_worker_tasks = None
        self._init_worker(config.pipeline_cls)

    def _init_worker(self, pipeline_cls):
        world_size = self.config.num_gpus

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Set OMP_NUM_THREADS to 1 if it is not set explicitly, avoids CPU
        # contention amongst the shards
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"

        # NOTE: The two following lines need adaption for multi-node
        assert world_size <= torch.cuda.device_count()

        # change addr for multi-node
        distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())

        if world_size == 1:
            self.workers = []
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_pipeline,
                        pipeline_cls=pipeline_cls,
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    ),
                )
                for rank in range(1, world_size)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_pipeline(
            pipeline_cls=pipeline_cls, distributed_init_method=distributed_init_method
        )

    # TODO: add more options here for pipeline, or wrap all options into config
    def _create_pipeline(self, pipeline_cls, rank=0, local_rank=0, distributed_init_method=None):
        videosys.initialize(rank=rank, world_size=self.config.num_gpus, init_method=distributed_init_method)

        pipeline = pipeline_cls(self.config)
        return pipeline

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        # Start the workers first.
        worker_outputs = [worker.execute_method(method, *args, **kwargs) for worker in self.workers]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output] + [output.get() for output in worker_outputs]

    def _driver_execute_model(self, *args, **kwargs):
        return self.driver_worker.generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._run_workers("generate", *args, **kwargs)[0]

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    def save_video(self, video, output_path):
        return self.driver_worker.save_video(video, output_path)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor", None)) is not None:
            worker_monitor.close()
        dist.destroy_process_group()

    def __del__(self):
        self.shutdown()

class VideoSysRayEngine:
    """
    this is partly inspired by vllm
    """

    def __init__(self, config):
        self.config = config
        self.parallel_worker_tasks = None
        placement_group = initialize_ray_cluster(config.ray_address, config.num_gpus)
        self._init_workers_ray(placement_group)

    def _init_workers_ray(self, placement_group,
                          **ray_remote_kwargs):
        num_gpus = 1

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker = None
        # The remaining workers are the actual ray actors.
        self.workers = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: List[List[RayWorkerWrapper]] = []

        # Create the workers.
        driver_ip = get_ip()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote()

            self.workers.append(worker)

        logger.debug("workers: %s", self.workers)

        worker_ips = [
            ray.get(worker.get_node_ip.remote())  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(worker):
            ip = ray.get(worker.get_node_ip.remote())
            return (ip != driver_ip, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        self.workers = sorted(self.workers, key=sort_by_driver_then_worker_ip)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids")

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration.")

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        init_dist_all_kwargs = [
            dict(
                rank=rank,
                world_size=len(self.workers),
                init_method=distributed_init_method,
            ) for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_dist", all_kwargs=init_dist_all_kwargs)

        init_pipeline_all_kwargs = [
            dict(
                pipeline_cls=self.config.pipeline_cls,
                config=self.config,
            ) for rank, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids)
        ]

        self._run_workers("init_pipeline", all_kwargs=init_pipeline_all_kwargs)

    def _run_workers(
        self,
        method: str,
        *args,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        count = len(self.workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0 
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)

        # Start the ray workers first.
        ray_workers = self.workers
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs


    def generate(self, *args, **kwargs):
        return self._run_workers("generate", *args, **kwargs)[0]
    
    async def generate_async(self, *args, **kwargs):
        return self._run_workers("generate", *args, **kwargs)[0]
    
    async def diffusion_step(self, *args, **kwargs):
        rank = kwargs.pop("rank", 0)
        await self.workers[rank].execute_method.remote("diffusion_step", *args, **kwargs)

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    def save_video(self, video, output_path):
        return self.workers[0].execute_method.remote("save_video", video, output_path)

    def shutdown(self) -> None:
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    def __del__(self):
        self.shutdown()
