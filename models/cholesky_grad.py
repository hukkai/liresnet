import torch


def orth(X):
    S = X @ X.mT
    eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3).detach()
    eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
    S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
    L = torch.linalg.cholesky(S)
    W = torch.linalg.solve_triangular(L, X, upper=False)
    return W


class CholeskyOrthfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        S = X @ X.mT
        eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3)
        eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
        S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
        L = torch.linalg.cholesky(S)
        W = torch.linalg.solve_triangular(L, X, upper=False)
        ctx.save_for_backward(W, L)
        return W

    @staticmethod
    def backward(ctx, grad_output):
        W, L = ctx.saved_tensors
        LmT = L.mT.contiguous()
        gB = torch.linalg.solve_triangular(LmT, grad_output, upper=True)
        gA = (-gB @ W.mT).tril()
        gS = (LmT @ gA).tril()
        gS = gS + gS.tril(-1).mT
        gS = torch.linalg.solve_triangular(LmT, gS, upper=True)
        gX = gS @ W + gB
        return gX


class CholeskyOrthfn_stable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        S = X @ X.mT
        eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3)
        eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
        S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
        L = torch.linalg.cholesky(S)
        W = torch.linalg.solve_triangular(L, X, upper=False)
        ctx.save_for_backward(X, W, L)
        return W

    @staticmethod
    def backward(ctx, grad_output):
        X, W, L = ctx.saved_tensors
        gB = torch.linalg.solve_triangular(L.mT, grad_output, upper=True)
        gA = (-gB @ W.mT).tril()
        gS = (L.mT @ gA).tril()
        gS = gS + gS.tril(-1).mT
        gS = torch.linalg.solve_triangular(L.mT, gS, upper=True)
        gS = torch.linalg.solve_triangular(L, gS, upper=False, left=False)
        gX = gS @ X + gB
        return gX


CholeskyOrth = CholeskyOrthfn.apply
