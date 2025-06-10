import torch


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer - optimized implementation.

    Forward pass is identity, backward pass multiplies gradient by -alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer class
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def update_alpha(self, alpha):
        """Update the alpha parameter for gradient reversal."""
        self.alpha = alpha