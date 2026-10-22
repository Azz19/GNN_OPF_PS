"""
Two processors that share an encoder/decoder, differing ONLY in depth:

  FixedDepthGNN : K weight-untied message-passing layers (the standard baseline)
  DEQGNN        : ONE weight-tied message-passing block iterated to a fixed
                  point z* = f(z*, x); trained by implicit differentiation.

The comparison isolates the research claim: iterating to equilibrium vs
iterating a fixed number of steps, at (near-)matched parameter count.

Message passing is implemented with plain PyTorch scatter-adds (index_add_)
so the scaffold needs no torch-geometric dependency.

DEQ backward pass: for z* = f(z*, u), the implicit function theorem gives
  dz*/du = (I - J_f)^-1 df/du.
We solve the adjoint system (I - J_f^T) w = dL/dz* by fixed-point iteration
(Neumann-style), the standard approach from Bai, Kolter & Koltun (2019)
reference implementations -- no Jacobian is ever materialised.
"""
import torch
import torch.nn as nn


class MPLayer(nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.Tanh())
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index[0], edge_index[1]
        m_in = torch.cat([h[src], edge_feat], dim=-1)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, self.msg(m_in))
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class Encoder(nn.Module):
    def __init__(self, node_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(node_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Predict (Vm, Va) per bus. Vm squashed into [0.85, 1.15] via sigmoid."""

    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Linear(hidden, 2)

    def forward(self, h):
        out = self.net(h)
        vm = 0.85 + 0.30 * torch.sigmoid(out[:, 0])
        va = out[:, 1]
        return torch.stack([vm, va], dim=-1)


class FixedDepthGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=64, depth=8):
        super().__init__()
        self.enc = Encoder(node_dim, hidden)
        self.layers = nn.ModuleList([MPLayer(hidden, edge_dim)
                                     for _ in range(depth)])
        self.dec = Decoder(hidden)

    def forward(self, x, edge_index, edge_feat):
        h = self.enc(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
        return self.dec(h)


# ----------------------------- DEQ layer ---------------------------------

class _DEQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, h0, u, edge_index, edge_feat, tol, max_iter):
        with torch.no_grad():
            z = h0
            for it in range(max_iter):
                z_new = block(z, u, edge_index, edge_feat)
                res = (z_new - z).norm().item()
                z = z_new
                if res < tol:
                    break
            ctx.last_iters = it + 1
            ctx.last_residual = res
        ctx.save_for_backward(z, u)
        ctx.block = block
        ctx.edge_index = edge_index
        ctx.edge_feat = edge_feat
        ctx.tol = tol
        ctx.max_iter = max_iter
        return z

    @staticmethod
    def backward(ctx, grad_out):
        z, u = ctx.saved_tensors
        block = ctx.block
        # Solve (I - J^T) w = grad_out by fixed-point iteration:
        #   w <- grad_out + J^T w,  J^T w via autograd vector-Jacobian product.
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            f = block(z_req, u.detach(), ctx.edge_index, ctx.edge_feat)
            w = grad_out.clone()
            for _ in range(ctx.max_iter):
                (jt_w,) = torch.autograd.grad(f, z_req, w, retain_graph=True)
                w_new = grad_out + jt_w
                if (w_new - w).norm() < ctx.tol:
                    w = w_new
                    break
                w = w_new
            # df/du and df/d(params) with vector w
            grads = torch.autograd.grad(f, [u.detach().requires_grad_(True)]
                                        + list(block.parameters()), w,
                                        retain_graph=False, allow_unused=True)
        grad_u = grads[0]
        param_grads = grads[1:]
        # stitch parameter gradients back
        for p, g in zip(block.parameters(), param_grads):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g
        return None, None, grad_u, None, None, None, None


class DEQGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=64, tol=1e-3, max_iter=50):
        super().__init__()
        self.enc = Encoder(node_dim, hidden)
        self.block = _TiedBlock(hidden, edge_dim)
        self.dec = Decoder(hidden)
        self.tol = tol
        self.max_iter = max_iter
        self.last_iters = None
        self.last_residual = None

    def forward(self, x, edge_index, edge_feat):
        u = self.enc(x)
        h0 = torch.zeros_like(u)
        z = _DEQFunction.apply(self.block, h0, u, edge_index, edge_feat,
                               self.tol, self.max_iter)
        return self.dec(z)

    @torch.no_grad()
    def solve_with_stats(self, x, edge_index, edge_feat):
        """Inference that also reports fixed-point iterations and residual --
        the DEQ-specific reporting the gap map calls for (G1/G9)."""
        u = self.enc(x)
        z = torch.zeros_like(u)
        for it in range(self.max_iter):
            z_new = self.block(z, u, edge_index, edge_feat)
            res = (z_new - z).norm().item()
            z = z_new
            if res < self.tol:
                break
        return self.dec(z), it + 1, res


class _TiedBlock(nn.Module):
    """One MP layer, weight-tied; input injection of the encoded features u
    at every iteration (standard DEQ input injection)."""

    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.layer = MPLayer(hidden, edge_dim)
        self.inject = nn.Linear(hidden, hidden)

    def forward(self, z, u, edge_index, edge_feat):
        return self.layer(z + self.inject(u), edge_index, edge_feat)
