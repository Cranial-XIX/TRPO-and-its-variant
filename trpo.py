import numpy as np
import torch

from torch.autograd import Variable
from utils import *


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(
    model, f, x,
    fullstep, expected_improve_rate,
    max_backtracks=10, accept_ratio=.1):

    fval = f(True).data
    #print("fval before", fval[0])
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        #print("a/e/r", actual_improve[0], expected_improve[0], ratio[0])

        if ratio[0] > accept_ratio and actual_improve[0] > 0:
        #    print("fval after", newfval[0])
            print "success, actual improve: ", actual_improve[0]
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, max_kl, damping, algo):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    
    if algo == 'trpo-kl':
        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

        stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        #print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(
            model, get_loss, prev_params,
            fullstep, neggdotstepdir / lm[0]
        )
        set_flat_params_to(model, new_params)

    elif algo == 'trpo-mse':
        # below is using mse constraint
        stepdir = -loss_grad

        mse_norm = torch.sum(loss_grad * loss_grad)
        lm = np.sqrt(mse_norm/max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(
            model, get_loss, prev_params, 
            fullstep, neggdotstepdir / lm
        )

        set_flat_params_to(model, new_params)

    else:
        # below is pure optimizing
        stepdir = -loss_grad
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        prev_params = get_flat_params_from(model)
        success, new_params = linesearch(
            model, get_loss, prev_params,
            stepdir, neggdotstepdir
        )

        set_flat_params_to(model, new_params)

    return
