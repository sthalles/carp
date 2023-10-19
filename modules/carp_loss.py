import torch
import torch.nn as nn
import torch.distributed as dist


class CARPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cluster_loss(p, q, EPS):
        # assert inputs.shape == targets.shape
        # assert inputs.requires_grad == True
        # assert targets.requires_grad == False

        loss = torch.einsum("knc,knc->kn", [p, q])
        loss = torch.clamp(loss, EPS, 1.0 - EPS)
        loss = -torch.log(loss).mean()
        return loss

    def forward(self, student_output, teacher_output):
        EPS = torch.finfo(student_output[0].dtype).eps
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                consistency += self.cluster_loss(
                    student_output[i], teacher_output[j], EPS)
                count += 1

        consistency /= count

        p = torch.cat(student_output, dim=1)
        q = torch.cat(teacher_output, dim=1)
        probs = torch.cat([p, q], dim=1)  # [N_GROUPS, 2*BS, DIM]
        probs = torch.transpose(probs, 0, 1)  # [2*BS, N_GROUPS, DIM]
        probs = AllGather.apply(probs)

        entropy = self.kl_div(torch.mean(probs, dim=0), EPS)
        return consistency, entropy

    @staticmethod
    def kl_div(p, EPS):
        return (
            torch.log(torch.tensor(
                p.shape[-1], dtype=p.dtype, device=p.device))
            + torch.sum(p * torch.log(torch.clamp(p, EPS, 1.0 - EPS)), axis=-1)
        ).mean()


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
                dist.is_available()
                and dist.is_initialized()
                and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x)
                       for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
                dist.is_available()
                and dist.is_initialized()
                and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * \
                (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads
