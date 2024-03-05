import torch


class Sigmoid(torch.autograd.Function):
    """The Sigmoid activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        neg_mask = data < 0
        pos_mask = ~neg_mask

        zs = torch.empty_like(data)
        zs[neg_mask] = data[neg_mask].exp()
        zs[pos_mask] = (-data[pos_mask]).exp()

        res = torch.ones_like(data)
        res[neg_mask] = zs[neg_mask]

        result = res / (1 + zs)

        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        (result,) = ctx.saved_tensors
        grad = result * (1 - result)
        return grad_output * grad


if __name__ == "__main__":
    # Sets the manual seed for reproducible experiments
    torch.manual_seed(0)

    sigmoid = Sigmoid.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    # `torch.autograd.gradcheck` takes a tuple of tensors as input, check if your gradient evaluated
    # with these tensors are close enough to numerical approximations and returns `True` if they all
    # verify this condition.
    if torch.autograd.gradcheck(sigmoid, data, eps=1e-8, atol=1e-7):  # type: ignore
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")
