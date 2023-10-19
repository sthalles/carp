import torch
import torch.nn as nn
import utils


class RandomPartition(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ncrops = args.ncrops
        self.n_prototypes = args.out_dim
        self.weights = torch.ones([args.out_dim,], dtype=torch.float)

    def forward(self, student_output, teacher_output, partition_size):

        student_out = student_output.chunk(self.ncrops)
        teacher_out = teacher_output.detach().chunk(2)

        number_of_partitions = self.n_prototypes // partition_size

        # logic for rangom partioning into subgroups
        if utils.is_dist_avail_and_initialized():

            if utils.get_rank() == 0:
                rand_cluster_indices = torch.multinomial(
                    self.weights,
                    number_of_partitions * partition_size,
                    replacement=False,
                ).cuda()
            else:
                rand_cluster_indices = torch.zeros(
                    (number_of_partitions * partition_size), dtype=torch.long
                ).cuda()

            torch.distributed.broadcast(rand_cluster_indices, src=0)
        else:
            rand_cluster_indices = torch.multinomial(
                self.weights,
                number_of_partitions * partition_size,
                replacement=False,
            ).cuda()

        split_cluster_ids = torch.stack(
            torch.split(rand_cluster_indices, partition_size)
        )

        probs_list = []
        for log_view in student_out:
            predictions_group = self.get_logits_group(
                log_view, split_cluster_ids, partition_size)
            probs_list.append(predictions_group)

        targets_list = []
        for tar_view in teacher_out:
            targets_group = self.get_logits_group(
                tar_view, split_cluster_ids, partition_size)
            targets_list.append(targets_group)

        return probs_list, targets_list

    def get_logits_group(self, logits, split_cluster_ids, partition_size):
        logits_group = logits[:, split_cluster_ids.flatten()]
        logits_group = logits_group.split(partition_size, dim=1)
        # out shape [N_BLOCKS, BS, BLOCK_SIZE]
        logits = torch.stack(logits_group, dim=0)
        probs = torch.softmax(logits, dim=-1)
        return probs
