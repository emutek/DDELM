import torch
from functools import partial
from typing import Callable

Tensor = torch.Tensor
dtype = torch.float64
ones = partial(torch.ones, dtype=dtype)
zeros = partial(torch.zeros, dtype=dtype)
empty = partial(torch.empty, dtype=dtype)
linspace = partial(torch.linspace, dtype=dtype)
eye = partial(torch.eye, dtype=dtype)

from mpi4py import MPI
comm = MPI.COMM_WORLD


def cg_distributed(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=lambda x: x,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  ):
  if x0 is None:
    x0 = zeros(b.shape)
  if maxiter is None:
    maxiter = problem_size * 10
    
  rsold, rsnew, pAp = empty([]), empty([]), empty([])

  x = x0
  r = b - A(x)
  z = M(r)
  p = z
  
  comm.Allreduce(r @ (z / multiplicity), rsold, op=MPI.SUM)

  atol = torch.maximum(tol * rsold ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))

  i = 0
  while i < maxiter and rsold ** .5 > atol:
    Ap = A(p)
    comm.Allreduce(p @ (Ap / multiplicity), pAp, op=MPI.SUM)
    alpha = rsold / pAp
    x = x + alpha * p
    r = r - alpha * Ap
    z = M(r)
    comm.Allreduce(r @ (z / multiplicity), rsnew, op=MPI.SUM)
    beta = rsnew / rsold
    p = z + beta * p
    rsold[None] = rsnew
    i += 1

  return x, (i, rsold ** .5)

def gmres_distributed(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=lambda x: x,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  restart: int=20,
  safety_eps: float=1e-40,
  mgs: bool=False
  ):
  if maxiter is None:
    maxiter = problem_size * 3
  if restart is None:
    restart = problem_size
  if restart > problem_size:
    restart = problem_size

  Q = empty(b.shape[0], restart)
  H = zeros(restart + 1, restart)
  H_gs = zeros(restart)

  r0_ss, v_ss = empty([]), empty([])

  if x0 is None:
    x0 = zeros(b.shape)
    r0 = b
    comm.Allreduce(r0 @ (r0 / multiplicity), r0_ss, op=MPI.SUM)
  else:
    r0 = b - M(A(x0))
    comm.Allreduce(r0 @ (r0 / multiplicity), r0_ss, op=MPI.SUM)
  atol = torch.maximum(tol * r0_ss ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))

  e1 = zeros(restart + 1)
  e1[0] = 1
  first_flag = True
  for i_iter in range(maxiter):
    p = ones(1) # the orthogonal complement of H

    if first_flag:
      first_flag = False
      v_ss[None] = r0_ss
    else:
      r0 = b - M(A(x0))
      comm.Allreduce(r0 @ (r0 / multiplicity), v_ss, op=MPI.SUM)

    v = r0
    v_l2 = v_ss ** .5
    r0_l2 = v_l2
    for i_restart in range(restart):
      Q[:, i_restart] = v / v_l2
      v = M(A(Q[:, i_restart]))

      if mgs:
        # modified Gram Schmidt
        for i_mgs in range(i_restart + 1):
          comm.Allreduce(Q[:, i_mgs] @ (v / multiplicity), H[i_mgs, i_restart], op=MPI.SUM)
          v -= Q[:, i_mgs] * H[i_mgs, i_restart]

      else:
        # no mod
        comm.Allreduce((v / multiplicity) @ Q[:, :i_restart+1], H_gs[:i_restart+1], op=MPI.SUM)
        H[:i_restart+1, i_restart] = H_gs[:i_restart+1]
        v -= Q[:, :i_restart+1] @ H[:i_restart+1, i_restart]

      comm.Allreduce(v @ (v / multiplicity), v_ss, op=MPI.SUM)
      v_l2 = v_ss ** .5
      H[i_restart + 1, i_restart] = v_l2

      w = p @ H[:i_restart+1, i_restart]
      p = torch.cat([v_l2 * p, -w.unsqueeze(0)])
      p = p / (H[i_restart+1, i_restart] ** 2 + w ** 2) ** .5
      
      res = p[0] * r0_l2
      if res < atol:
        break
    
    e1_approx = r0_l2 * e1[:i_restart+2] - res * p
    if H[i_restart+1, i_restart] == 0:
      H[i_restart+1, i_restart] = safety_eps
    y = torch.linalg.solve_triangular(H[1:i_restart+2, :i_restart+1], e1_approx[1:].unsqueeze(1), upper=True).reshape(-1)

    x0 += torch.einsum('ji, i->j', Q[:, :i_restart+1], y)
    if res < atol:
      break
    
  return x0, (i_iter+1, i_restart+1, res)