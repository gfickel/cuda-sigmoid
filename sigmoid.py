import torch


class Sigmoid(torch.autograd.Function):
    """The Sigmoid activation function."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        out_tensor = torch.empty_like(input)
        positive_mask = input >= 0
        out_tensor[positive_mask] = 1. / (1. + torch.exp(-input[positive_mask]))
        out_tensor[~positive_mask] = torch.exp(input[~positive_mask]) / (1. + torch.exp(input[~positive_mask]))
        
        ctx.save_for_backward(out_tensor)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        (result,) = ctx.saved_tensors
        grad = result * (1 - result)
        return grad_output * grad


if __name__ == "__main__":
    torch.manual_seed(42)

    sigmoid = Sigmoid.apply
    data = torch.randn(4, dtype=torch.double, requires_grad=True)

    if torch.autograd.gradcheck(sigmoid, data, eps=1e-8, atol=1e-7):
        print('gradcheck successful :D')
    else:
        print('gradcheck unsuccessful :D')
