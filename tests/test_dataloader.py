import colossalai
import torch
import torch.distributed as dist
from colossalai.testing import parameterize
from colossalai.utils import get_current_device
from torchvision import transforms
from torchvision.datasets import CIFAR10

from opendit.utils.data_utils import center_crop_arr, prepare_dataloader
from opendit.utils.pg_utils import ProcessGroupManager


@parameterize("batch_size", [2])
@parameterize("num_workers", [4])
@parameterize("sequence_parallel_size", [2, 4])
@parameterize("image_size", [256])
def run_dataloader_test(batch_size, num_workers, sequence_parallel_size, image_size, data_path="../datasets"):
    sp_size = sequence_parallel_size
    dp_size = dist.get_world_size() // sp_size
    pg_manager = ProcessGroupManager(dp_size, sp_size, dp_axis=0, sp_axis=1)
    device = get_current_device()

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = CIFAR10(data_path, transform=transform, download=True)
    dataloader = prepare_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        pg_manager=pg_manager,
    )
    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)
    x = x.to(device)
    y = y.to(device)

    x_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    y_list = [torch.empty_like(y) for _ in range(dist.get_world_size())]

    dist.all_gather(x_list, x)
    dist.all_gather(y_list, y)

    sp_group_ranks = pg_manager.get_ranks_in_group(pg_manager.sp_group)
    dp_group_ranks = pg_manager.get_ranks_in_group(pg_manager.dp_group)

    for rank in sp_group_ranks:
        if rank != dist.get_rank():
            assert torch.allclose(
                x_list[rank], x_list[dist.get_rank()]
            ), f"x in rank {rank} and {dist.get_rank()} are not equal in the same sequence parallel group."
            assert torch.allclose(
                y_list[rank], y_list[dist.get_rank()]
            ), f"y in rank {rank} and {dist.get_rank()} are not equal in the same sequence parallel group."

    for rank in dp_group_ranks:
        if rank != dist.get_rank():
            assert not torch.allclose(
                x_list[rank], x_list[dist.get_rank()]
            ), f"x in rank {rank} and {dist.get_rank()} are equal in the same data parallel group."
            assert not torch.allclose(
                y_list[rank], y_list[dist.get_rank()]
            ), f"y in rank {rank} and {dist.get_rank()} are equal in the same data parallel group."

    if dist.get_rank() == 0:
        print(f"DataLoader test (world size {dist.get_world_size()}, sequence parallel size {sp_size}) passed!")


def check_dataloader():
    colossalai.launch_from_torch({})
    run_dataloader_test()


if __name__ == "__main__":
    check_dataloader()
