import torch
from typing import Callable, Union, NamedTuple
from abc import ABC, abstractmethod
from torch.func import vmap, vjp, jvp, jacrev, hessian, jacfwd
from functools import partial, wraps
from inspect import signature

Tensor = torch.Tensor
dtype = torch.float64
ones = partial(torch.ones, dtype=dtype)
zeros = partial(torch.zeros, dtype=dtype)
empty = partial(torch.empty, dtype=dtype)
linspace = partial(torch.linspace, dtype=dtype)
eye = partial(torch.eye, dtype=dtype)

import runtime_constants
chunk_size = runtime_constants.chunk_size

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from linear_solvers import cg_distributed, gmres_distributed

# basically direct lift of answer in stackoverflow
# https://stackoverflow.com/questions/11065419/unpacking-keyword-arguments-but-only-the-ones-that-match-the-function
def nonkwargs_to_kwargs(foo):
  if any([arg.kind == arg.VAR_KEYWORD for arg in signature(foo).parameters.values()]):
    return foo
  @wraps(foo)
  def wrapped_foo(*args, **kwargs):
    subset = dict((key, kwargs[key]) for key in kwargs if key in signature(foo).parameters)
    return foo(*args, **subset)
  return wrapped_foo

def act_last(f: Callable, where_x: Union[int, list[int]]=0, chunk_size: int=None):
  in_dims = [None for v in signature(f).parameters.values() if v.kind is v.POSITIONAL_OR_KEYWORD]
  if isinstance(where_x, int):
    in_dims[where_x] = 0
  else:
    for i in where_x:
      in_dims[i] = 0
  in_dims = tuple(in_dims)
  vfoo = vmap(f, in_dims=in_dims, chunk_size=chunk_size)
  @wraps(f)
  def wrapper(*args, **kwargs):
    if isinstance(where_x, int):
      shape = args[where_x].shape
      args = list(args)
      args[where_x] = args[where_x].reshape(-1, shape[-1])
    else:
      shape = args[where_x[0]].shape
      args = list(args)
      for i in where_x:
        args[i] = args[i].reshape(-1, shape[-1])
    rax = vfoo(*args, **kwargs)
    return rax.reshape(shape[:-1] + rax.shape[1:])
  return wrapper

class Interface(NamedTuple):
  idcs: Tensor
  rank: int
  flux_info: Tensor

class Grid(NamedTuple):
  int: Tensor
  bou: Tensor
  gam: Tensor
  prim: list[Interface]
  prim_to_glob: Tensor
  prim_multiplicity: Tensor

class YKDD(ABC):
  def __init__(self, **kwargs) -> None:
    self.act: Callable[[Tensor], Tensor] = torch.tanh

    self.define_problem_params(**kwargs)

  @nonkwargs_to_kwargs
  def define_problem_params(
    self, *,
    target_rhs: Callable[[Tensor], Tensor]=None,
    target_bou: Callable[[Tensor], Tensor]=None
  ) -> None:
    self.target_rhs = target_rhs
    self.target_bou = target_bou

  def net_x(self, x: Tensor, theta) -> Tensor:
    return self.act(x @ theta['w'] + theta['b'])
  
  def net_xa(self, x: Tensor, a: Tensor, theta) -> Tensor:
    return self.net_x(x, theta) @ a
  
  @partial(act_last, where_x=1, chunk_size=chunk_size)
  def dx_net_x(self, x: Tensor, theta) -> Tensor:
    return jacfwd(self.net_x, argnums=0)(x, theta)
  
  def dx_net_xa(self, x: Tensor, a: Tensor, theta) -> Tensor:
    return torch.einsum('...Md, M->...d', self.dx_net_x(x, theta), a)
  
  @abstractmethod
  def Lnet_x_single(self, x: Tensor, theta) -> Tensor:
    return empty(x.shape[:-1])

  @partial(act_last, where_x=1, chunk_size=chunk_size)
  def Lnet_x(self, x: Tensor, theta) -> Tensor:
    return self.Lnet_x_single(x, theta)

  def Lnet_xa(self, x: Tensor, a: Tensor, theta) -> Tensor:
    return self.Lnet_x(x, theta) @ a
  
  @abstractmethod
  def fluxnet_x_single(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return empty(x.shape[:-1])
  
  @partial(act_last, where_x=[1, 3], chunk_size=chunk_size)
  def fluxnet_x(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return self.fluxnet_x_single(x, theta, flux_info)
  
  def get_grid(self, *, n):
    """Points for domain decomposition in standard rectangular 2D grid."""
    N = int(size ** .5)
    i, j = rank // N, rank % N
    linx = linspace(0, 1, n+1) / N
    grid = torch.cartesian_prod(linx + i / N, linx + j / N).reshape(n+1, n+1, 2)

    x_int = grid[1:-1, 1:-1].reshape(-1, 2)
    x_bou = torch.cat(
      [
        grid[0, :-1],
        grid[:-1, -1],
        grid[-1, 1:].flip(0),
        grid[1:, 0].flip(0)
      ]
    )

    empty_idx = torch.arange(0)
    ind_bou_idcs = torch.unique(
      torch.cat(
        [
          torch.arange(n+1) if i == 0 else empty_idx,
          torch.arange(n, 2*n+1) if j == N-1 else empty_idx,
          torch.arange(2*n, 3*n+1) if i == N-1 else empty_idx,
          torch.arange(3*n, 4*n) if j == 0 else empty_idx,
          torch.arange(1) if j == 0 else empty_idx
        ]
      )
    )
    x_ind_bou = x_bou[ind_bou_idcs]

    gam_mask = torch.ones(4*n, dtype=torch.bool)
    gam_mask[ind_bou_idcs] = False
    bou_to_gam = gam_mask.cumsum(0) - 1
    x_gam = x_bou[gam_mask]

    # x_dual = []
    x_prim = []
    if i > 0:
      # up
      idcs_to_x_bou = torch.arange(0 if j > 0 else 1, n+1 if j < N-1 else n)
      x_prim.append(
        Interface(
          bou_to_gam[idcs_to_x_bou], # idcs to x_gam
          rank - N, # rank sharing these points
          torch.tensor([-1, 0], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2) # information to get flux; in this case the normal vector
        )
      )
    if j < N-1:
      # right
      idcs_to_x_bou = torch.arange(n if i > 0 else n+1, 2*n+1 if i < N-1 else 2*n)
      x_prim.append(
        Interface(
          bou_to_gam[idcs_to_x_bou],
          rank + 1,
          torch.tensor([0, 1], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2)
        )
      )
    if i < N-1:
      # down
      idcs_to_x_bou = torch.arange(3*n if j > 0 else 3*n-1, 2*n-1 if j < N-1 else 2*n, -1)
      x_prim.append(
        Interface(
          bou_to_gam[idcs_to_x_bou], # flip direction to match neighbor's up
          rank + N,
          torch.tensor([1, 0], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2)
        )
      )
    if j > 0:
      # left
      idcs_to_x_bou = torch.arange(0 if i > 0 else -1, -n-1 if i < N-1 else -n, -1)
      x_prim.append(
        Interface(
          bou_to_gam[idcs_to_x_bou],
          rank - 1,
          torch.tensor([0, -1], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2)
        )
      )
    # corners
    # up left
    if i > 0 and j > 0:
      x_prim.append(
        Interface(
          bou_to_gam[0:1],
          rank - N - 1,
          # torch.tensor([-1, -1], dtype=dtype).expand(1, 2)
          None
        )
      )
    # up right
    if i > 0 and j < N-1:
      x_prim.append(
        Interface(
          bou_to_gam[n:n+1],
          rank - N + 1,
          # torch.tensor([-1, 1], dtype=dtype).expand(1, 2)
          None
      )
    )
    # down right
    if i < N-1 and j < N-1:
      x_prim.append(
        Interface(
          bou_to_gam[2*n:2*n+1],
          rank + N + 1,
          # torch.tensor([1, 1], dtype=dtype).expand(1, 2)
          None
      )
    )
    # down left
    if i < N-1 and j > 0:
      x_prim.append(
        Interface(
          bou_to_gam[3*n:3*n+1],
          rank + N - 1,
          # torch.tensor([1, -1], dtype=dtype).expand(1, 2)
          None
      )
    )

    #   y0  ------------------->  1
    #   -----------------------------
    # x |  0   |  1   |  2   |  3   |
    # 0 |      0      1      2      |
    #   |--12--o--15--o--18--o--21--|
    # | |  4   |  5   |  6   |  7   |  
    # | |      3      4      5      |
    # | |--13--o--16--o--19--o--22--|
    # | |  8   |  9   |  10  |  11  |
    # | |      6      7      8      |
    # V |--14--o--17--o--20--o--23--|
    #   |  12  |  13  |  14  |  15  |
    # 1 |      9     10     11      |
    #   -----------------------------
    # the o are numbered
    # 24  25  26
    # 27  28  29
    # 30  31  32

    one_idx = torch.arange(1,2)
    prim_to_glob = torch.cat(
      [
        one_idx * (2*(n-1)*N*(N-1) + (N-1)*(i-1) + j-1) if i > 0 and j > 0 else empty_idx, # up left corner
        torch.arange(0, n-1) + (n-1)*(N*(N-1) + (N-1)*j + i-1) if i > 0 else empty_idx, # up
        one_idx * (2*(n-1)*N*(N-1) + (N-1)*(i-1) + j) if i > 0 and j < N-1 else empty_idx, # up right corner
        torch.arange(0, n-1) + (n-1)*((N-1)*i + j) if j < N-1 else empty_idx, # right
        one_idx * (2*(n-1)*N*(N-1) + (N-1)*i + j) if i < N-1 and j < N-1 else empty_idx, # down right corner
        torch.arange(n-2, -1, -1) + (n-1)*(N*(N-1) + (N-1)*j + i) if i < N-1 else empty_idx, # down, flip direction
        one_idx * (2*(n-1)*N*(N-1) + (N-1)*i + j-1) if i < N-1 and j > 0 else empty_idx, # down left corner
        torch.arange(n-2, -1, -1) + (n-1)*((N-1)*i + j-1) if j > 0 else empty_idx, # left
      ]
    )

    prim_multiplicity = torch.cat(
      [
        one_idx * 4 if i > 0 and j > 0 else empty_idx, # up left corner
        ones(n-1) * 2 if i > 0 else empty_idx, # up
        one_idx * 4 if i > 0 and j < N-1 else empty_idx, # up right corner
        ones(n-1) * 2 if j < N-1 else empty_idx, # right
        one_idx * 4 if i < N-1 and j < N-1 else empty_idx, # down right corner
        ones(n-1) * 2 if i < N-1 else empty_idx, # down
        one_idx * 4 if i < N-1 and j > 0 else empty_idx, # down left corner
        ones(n-1) * 2 if j > 0 else empty_idx, # left
      ]
    )

    return Grid(x_int, x_ind_bou, x_gam, x_prim, prim_to_glob, prim_multiplicity)
  
  def get_K(self, grid: Grid, theta) -> tuple[Tensor, int, Callable[[Tensor], Tensor]]:
    K = torch.cat(
      [
        self.Lnet_x(grid.int, theta),
        self.net_x(grid.bou, theta),
        self.net_x(grid.gam, theta)
      ]
    )

    def gam_idcs_to_prim_idcs(idcs: Tensor) -> Tensor:
      return idcs
    
    return K, grid.gam.shape[0], gam_idcs_to_prim_idcs
  
  def get_rhs(self, grid: Grid) -> Tensor:
    return torch.cat(
      [
        self.target_rhs(grid.int),
        self.target_bou(grid.bou),
        zeros(grid.gam.shape[0])
      ]
    )
  
  def empty_flux(self, theta) -> Tensor:
    return empty(0, theta['b'].shape[0])
  
  def get_A(self, grid: Grid, theta) -> tuple[Tensor, list[int, int]]:
    idx = 0
    idcs = []
    empty_flux = self.empty_flux(theta)
    A = [empty_flux]
    for interface in grid.prim:
      if interface.flux_info is not None:
        A.append(self.fluxnet_x(grid.gam[interface.idcs], theta, interface.flux_info))
      else:
        A.append(empty_flux)
      idcs.append((idx, idx + A[-1].shape[0]))
      idx = idcs[-1][1]
    return torch.cat(A), idcs
  
  def get_B_flux(self, grid: Grid) -> tuple[Tensor, list[int, int]]:
    idx = 0
    idcs = []
    empty_flux = empty(0, grid.gam.shape[0])
    A = [empty_flux]
    for interface in grid.prim:
      if interface.flux_info is not None:
        B = zeros(interface.flux_info.shape[0], grid.gam.shape[0])
        B[torch.arange(B.shape[0]), interface.idcs] = interface.flux_info.sum(-1)
        A.append(B)
      else:
        A.append(empty_flux)
      idcs.append((idx, idx + A[-1].shape[0]))
      idx = idcs[-1][1]
    return torch.cat(A), idcs
  
  def get_multiplicity(self, grid: Grid) -> Tensor:
    return grid.prim_multiplicity
  
  def grid_communication(self, grid: Grid, message: list[Tensor], postbox: list[Tensor]):
    send_reqs = []
    recv_reqs = []
    for (interface, m, p) in zip(grid.prim, message, postbox):
      send_reqs.append(comm.Isend(m, dest=interface.rank))
      recv_reqs.append(comm.Irecv(p, source=interface.rank))
    
    for req in recv_reqs:
      req.wait()
    
    for req in send_reqs:
      req.wait()

    return
  
  def A_comms(self, grid: Grid, A: Tensor, A_idcs: list[tuple[int, int]]) -> Tensor:
    A_recv = empty(A.shape)
    message = [A[i:j] for i, j in A_idcs]
    postbox = [A_recv[i:j] for i, j in A_idcs]
    self.grid_communication(grid, message, postbox)
    return A + A_recv
  
  def BT_comms(self, grid: Grid, rbx: Tensor, BT_idcs: list[Tensor]) -> Tensor:
    message = [rbx[idcs] for idcs in BT_idcs]
    postbox = [empty(m.shape) for m in message]
    self.grid_communication(grid, message, postbox)
    rax = rbx.clone()
    for post, idcs in zip(postbox, BT_idcs):
      rax[idcs] += post
    return rax
  
  def ykdd(self, problem_size: int, grid: Grid, theta, *, lin_solver='cg', tol=1e-9, atol=0, maxiter=None, restart=None):
    K, gam_K_size, gam_to_prim = self.get_K(grid, theta)
    A, A_idcs = self.get_A(grid, theta)
    B = zeros(K.shape[0], gam_K_size)
    B[K.shape[0]-gam_K_size:, :] = -eye(gam_K_size)
    f = self.get_rhs(grid)
    multiplicity = self.get_multiplicity(grid)
  
    U, S, Vh = torch.linalg.svd(K, full_matrices=False)
    rcond = torch.finfo(K.dtype).eps * max(*K.shape)
    mask = S >= torch.tensor(rcond, dtype=K.dtype) * S[0]
    safe_idx = mask.sum()
    U, S, Vh = U[:, :safe_idx], S[:safe_idx], Vh[:safe_idx]
    S_inv = 1 / S
    UTB = U.T @ B
    AKpB = A @ (Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTB)))
    BTUUTB = UTB.T @ UTB

    BT_idcs = [gam_to_prim(interface.idcs) for interface in grid.prim]

    def apply_mat(x: Tensor) -> Tensor:
      rax1 = x # BTBu
      AKpBx = self.A_comms(grid, AKpB @ x, A_idcs)

      rax2 = AKpB.T @ AKpBx

      rax3 = BTUUTB @ x

      rax = self.BT_comms(grid, rax1 + rax2 - rax3, BT_idcs)
      return rax

    rhs1 = B.T @ f
    UTf = U.T @ f
    AKpf = A @ (Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTf)))
    AKpf = self.A_comms(grid, AKpf, A_idcs)
    rhs2 = AKpB.T @ AKpf
    rhs3 = UTB.T @ UTf

    rhs = rhs1 + rhs2 - rhs3
    rhs = self.BT_comms(grid, rhs, BT_idcs)

    if type(lin_solver) is str:
      if lin_solver == 'cg':  
        mu, info = cg_distributed(multiplicity, problem_size, apply_mat, rhs, tol=tol, atol=atol, maxiter=maxiter)
      elif lin_solver == 'gmres':
        mu, info = gmres_distributed(multiplicity, problem_size, apply_mat, rhs, tol=tol, atol=atol, maxiter=maxiter, restart=restart)

    else:
      mu, info = lin_solver(multiplicity, problem_size, apply_mat, rhs)

    a = Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTf - UTB @ mu))

    return a, info

  def elm(self, problem_size: int, grid: Grid, theta, *, driver='gelsd', **_):
    K, gam_K_size, gam_to_prim = self.get_K(grid, theta)
    f = self.get_rhs(grid)
    a = torch.linalg.lstsq(K, f, driver=driver).solution
    return a, (0, 0)
  
  def compute_errors(
      self, grid: Grid, a: Tensor, theta,
      target_exact: Callable[[Tensor], Tensor],
      target_exact_grad: Callable[[Tensor], Tensor]
    ):
    l2_points = torch.cat([grid.int, grid.bou, grid.gam])
    y = self.net_xa(l2_points, a, theta)
    y_exact = target_exact(l2_points)
    y_exact_ss = torch.square(y_exact).sum()
    y_error_ss = torch.square(y - y_exact).sum()

    h1_points = torch.cat([grid.int, grid.gam])
    y_grad = self.dx_net_xa(h1_points, a, theta)
    y_exact_grad = target_exact_grad(h1_points)
    y_exact_grad_ss = torch.square(y_exact_grad).sum()
    y_grad_error_ss = torch.square(y_grad - y_exact_grad).sum()

    mu = self.net_xa(grid.gam, a, theta)
    mu_exact = target_exact(grid.gam)
    mu_error_ss = (torch.square(mu - mu_exact) / grid.prim_multiplicity).sum()
    mu_n = (1 / grid.prim_multiplicity).sum()

    res = self.Lnet_xa(grid.int, a, theta) - self.target_rhs(grid.int)
    res_ss = torch.square(res).sum()
    res_n = grid.int.shape[0]

    buffer_size = 64
    packet = torch.tensor(
      [y_exact_ss, y_error_ss, y_exact_grad_ss, y_grad_error_ss, mu_error_ss, mu_n, res_ss, res_n], dtype=dtype
    )
    packet = torch.cat([packet, empty(buffer_size - packet.shape[0])])
    packet_recv = empty(buffer_size)
    comm.Reduce(packet, packet_recv, op=MPI.SUM, root=0)
    if rank == 0:
      print(
        f'l2 rel error: {(packet_recv[1] / packet_recv[0])**.5 : .4e}\n'
        f'h1 rel error: {((packet_recv[1] + packet_recv[3]) / (packet_recv[0] + packet_recv[2]))**.5 : .4e}\n'
        f'l2 error mu: {(packet_recv[4] / packet_recv[5])**.5 : .4e}\n'
        f'l2 error residual: {(packet_recv[6] / packet_recv[7])**.5 : .4e}'
      )
