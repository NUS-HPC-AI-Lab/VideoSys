# adapted from vllm
# https://github.com/vllm-project/vllm/blob/main/vllm/executor/ray_utils.py
import os
import time
import socket
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from videosys.utils.logging import create_logger
import torch.distributed as dist

import videosys

import ray
from ray.util import placement_group_table
from ray.util.placement_group import PlacementGroup
try:
    from ray._private.state import available_resources_per_node
except ImportError:
    # Ray 2.9.x doesn't expose `available_resources_per_node`
    from ray._private.state import state as _state
    available_resources_per_node = _state._available_resources_per_node

CYAN = "\033[1;36m"
PG_WAIT_TIMEOUT = 1800
logger = create_logger()

def get_ip() -> str:
    host_ip = ""
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default.",
        stacklevel=2)
    return "0.0.0.0"

def _verify_bundles(placement_group: "PlacementGroup"):
    """Verify a given placement group has bundles located in the right place.

    There are 2 rules.
    - Warn if all tensor parallel workers cannot fit in a single node.
    - Fail if driver node is not included in a placement group.
    """
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray.")
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (e.g., {"GPU": 1})
    bundles = pg_data["bundles"]
    # node_id -> List of bundle (e.g., {"GPU": 1})
    node_id_to_bundle: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
    driver_node_id = ray.get_runtime_context().get_node_id()

    if driver_node_id not in node_id_to_bundle:
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` to see if you have available GPUs in a node "
            f"{driver_node_id} before starting an VideoSys engine.")

def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check "
            "`ray status` to see if you have enough resources.",
            int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError(
            "Cannot provide a placement group of "
            f"{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See "
            "`ray status` to make sure the cluster has enough resources."
        ) from None


def initialize_ray_cluster(
    ray_address: Optional[str] = None,
    world_size: Optional[int] = None,
):
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    # Connect to a ray cluster.
    ray.init(address=ray_address, ignore_reinit_error=True)

    device_str = "GPU"
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 "
                    f"{device_str}.")
            if bundle_devices:
                device_bundles += 1
        if world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group."
                f"Required number of devices: {world_size}. "
                f"Total number of devices: {device_bundles}.")
    else:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if world_size > num_devices_in_cluster:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group.")
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{
            device_str: 1.0
        } for _ in range(world_size)])
        
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. VideoSys engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")
        # This way, at least bundle is required to be created in a current
        # node.
        placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group)
    return current_placement_group

class RayWorkerWrapper:

    def __init__(self, *args, **kwargs) -> None:
        self.worker = None

    def init_pipeline(self, pipeline_cls, config):
        pipeline = pipeline_cls(config)
        self.worker = pipeline

    def get_node_ip(self) -> str:
        return get_ip()

    def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        return node_id, gpu_ids
    
    def init_dist(self, rank: int, world_size: int, init_method: str):
        videosys.initialize(rank=rank, world_size=world_size, init_method=init_method)

    def execute_method(self, method, *args, **kwargs):
        try:
            target = self if self.worker is None else self.worker
            executor = getattr(target, method)
            return executor(*args, **kwargs)
        except Exception as e:
            msg = (f"Error executing method {method}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e
    
    def diffusion_step(self, *args, **kwargs):
        return self.worker.transformer(*args, **kwargs)

    def override_env_vars(self, vars: Dict[str, str]):
        os.environ.update(vars)