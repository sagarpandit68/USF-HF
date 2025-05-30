#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import scipy
from scipy.optimize import minimize, NonlinearConstraint, BFGS

import jax
from jax import lax
import jax.numpy as jnp
from jax import grad, jit, random
from pyscf import gto, scf


# Define molecule and basis set
mol = gto.Mole()
mol = gto.Mole()
mol.atom = [
    # ["C", (-4.311, 2.521, -0.004)],
    # ["C", (-3.416, 3.749, 0.023)],
    # ["H", (-4.217, 1.948, 0.924)],
    # ["H", (-4.054, 1.865, -0.841)],
    # ["H", (-5.359, 2.818, -0.113)],
    # ["C", (-1.944, 3.366, 0.164)],
    # ["H", (-3.716, 4.394, 0.857)],
    # ["H", (-3.563, 4.323, -0.900)],
    # ["C", (-1.054, 4.595, 0.249)],
    # ["H", (-1.635, 2.756, -0.693)],
    # ["H", (-1.801, 2.757, 1.064)],
    # ["H", (-1.146, 5.211, -0.651)],
    # ["H", (-0.005, 4.298, 0.351)],
    # ["H", (-1.317, 5.211, 1.115)]
    
    # ["C", (-4.301, 2.339, 0.081)],
    # ["C", (-3.707, 3.724, -0.107)],
    # ["H", (-4.018, 1.919, 1.052)],
    # ["H", (-3.958, 1.655, -0.702)],
    # ["H", (-5.394, 2.385, 0.037)],
    # ["C", (-2.189, 3.697, -0.049)],
    # ["H", (-4.090, 4.394, 0.670)],
    # ["H", (-4.028, 4.130, -1.073)],
    # ["H", (-1.841, 3.323, 0.919)],
    # ["H", (-1.786, 4.705, -0.189)],
    # ["H", (-1.778, 3.054, -0.834)]

    # ['O', (0.0, 0.0, 0.0)],
    # ['H', (0.0, -0.757, 0.587)],
    # ['H', (0.0, 0.757, 0.587)] 
    # ['O', (3.0, 0.0, 0.0)],
    # ['H', (3.0, -0.757, 0.587)],                                                                                                                       
    # ['H', (3.0, 0.757, 0)]

    # ['O', (0.0, 0.0, 0.0)],
    # ['H', (0.0, -0.757, 0.587)],
    # ['H', (0.0, 0.757, 0.587)]
    
    # ['C', (0.000000, 0.000000, 0.000000)],
    # ['H', (0.000000, 0.000000, 1.089000)],
    # ['H', (1.026719, 0.000000, -0.363000)],
    # ['H', (-0.513360, -0.889165, -0.363000)],
    # ['F', (-0.513360, 0.889165, -0.363000)]

    # ['N', (0.000000, 0.000000, 0.000000)],
    # ['H', (0.937700, 0.000000, -0.381600)],
    # ['H', (-0.468850, 0.812990, -0.381600)],
    # ['H', (-0.468850, -0.812990, -0.381600)]

    # ["C", (-2.818, 1.201, -0.000)],
    # ["H", (-3.183, 1.602, 0.948)],
    # ["H", (-3.182, 0.179, -0.126)],
    # ["H", (-3.183, 1.821, -0.822)],
    # ["H", (-1.726, 1.201, -0.000)]

    # ['C', (0.000000, 0.000000, -0.601100)],
    # ['C', (0.000000, 0.000000, 0.601100)],
    # ['H', (0.000000, 0.000000, -1.664500)],
    # ['H', (0.000000, 0.000000, 1.664500)]

    # ['C', (0.000000, 0.000000, 0.669500)],
    # ['C', (0.000000, 0.000000, -0.669500)],
    # ['H', (0.923000, 0.000000, 1.231000)],
    # ['H', (-0.923000, 0.000000, 1.231000)],
    # ['H', (0.923000, 0.000000, -1.231000)],
    # ['H', (-0.923000, 0.000000, -1.231000)]

    ['C', (0.000000, 0.000000, 0.756400)],
    ['C', (0.000000, 0.000000, -0.756400)],
    ['H', (1.026719, 0.000000, 1.150700)],
    ['H', (-0.513360, -0.889165, 1.150700)],
    ['H', (-0.513360, 0.889165, 1.150700)],
    ['H', (1.026719, 0.000000, -1.150700)],
    ['H', (-0.513360, -0.889165, -1.150700)],
    ['H', (-0.513360, 0.889165, -1.150700)]

    # ['C', (0.000000, 0.000000, 0.000000)],
    # ['O', (0.000000, 0.000000, 1.208000)],
    # ['H', (0.945221, 0.000000, -0.340000)],
    # ['H', (-0.945221, 0.000000, -0.340000)]
]
mol.unit = "angstrom"
mol.basis = 'cc-pVDZ'
mol.build()

# SCF calculation
mf = scf.RHF(mol)
energy = mf.kernel()
ao = mol.nao
nuclear_repulsion_energy = mol.energy_nuc()
print(f"Nuclear Repulsion Energy: {nuclear_repulsion_energy:.6f} Hartree")
print(f"Electron energy: {energy - nuclear_repulsion_energy:.6f} Hartree")
print("Number of basis functions:", ao)

# Compute one-electron integrals (overlap matrix, kinetic energy matrix, etc.)
S = mol.intor('int1e_ovlp')
T = mol.intor('int1e_kin')
V = mol.intor('int1e_nuc')
G = mol.intor('int2e')

H = T+V


times = []

for _ in range(10):
    start = time.time()
    # RHF energy calculation by PySCF
    mf = scf.RHF(mol)
    mf.scf()
    elapsed_time = time.time() - start
    print("SCF: {:.4f} seconds".format(elapsed_time))
    times.append(elapsed_time)
    
average_time = sum(times) / len(times)
print(f"Average SCF time: {average_time} seconds")


# Convert tensors to float32 JAX arrays
S = jnp.array(S, dtype=jnp.float32)
T = jnp.array(T, dtype=jnp.float32)
V = jnp.array(V, dtype=jnp.float32)
H = jnp.array(H, dtype=jnp.float32)
G = jnp.array(G, dtype=jnp.float32)
nuclear_repulsion_energy = jnp.float32(nuclear_repulsion_energy)


# Constants
Ne = mol.nelectron
occ_o = Ne // 2
num_repeats = 10
times = []

# Define the orthogonality constraint
@jax.jit
def orthogonality(x, S):
    C = jnp.reshape(x, (ao, occ_o))
    temp = jnp.matmul(S, C)
    temp2 = jnp.matmul(C.T, temp)
    return jnp.linalg.norm(temp2 - jnp.eye(occ_o))


# Compute the energy
@jax.jit
def compute_energy(x, JK, H, S):
    C = jnp.reshape(x, (ao, occ_o))
    P = 2 * jnp.matmul(C, C.T)
    G = jnp.einsum('lk,ijkl->ij', P, JK) - 0.5 * jnp.einsum('lk,ilkj->ij', P, JK)
    e1 = jnp.einsum('ji,ij->', P, H)
    e2 = jnp.einsum('ji,ij->', P, G)
    energy = 0.5 * (2 * e1 + e2)
    return energy

# Compute the loss
@jax.jit
def compute_loss(x, JK, H, S, mu, lam):
    energy = compute_energy(x, JK, H, S)
    h = orthogonality(x, S)
    loss = energy + mu * h ** 2 + lam * h
    return loss

# Compute the gradient
@jax.jit
def compute_gradient(x, JK, H, S, mu, lam):
    grad_loss = jax.grad(compute_loss)
    return grad_loss(x, JK, H, S, mu, lam)


def armijo_line_search(x_curr, x_next, loss_curr, loss_next, d_curr, factor, alpha, JK, H, S, mu, lam):

    state = {
        'x': x_next,
        'loss': loss_next,
        'alpha': alpha,
    }

    def cond_fn(state):
        return state['loss'] > loss_curr + state['alpha']*factor

    def body_fn(state):
        x_next = x_curr + state['alpha'] * d_curr
        loss_next = compute_loss(x_next, JK, H, S, mu, lam)
        alpha = 0.5 * state['alpha']
        return {
            'x': x_next,
            'loss': loss_next,
            'alpha': alpha
        }

    final_state = lax.while_loop(cond_fn, body_fn, state)
    return final_state['x'], final_state['loss'], final_state['alpha']

@jax.jit
def run_secant_phase(x_curr, constraint, adapt_tol, koeff, mu, lam, JK, H, S):
    
    def inner_cond(inner_state):
        grad_norm = jnp.linalg.norm(inner_state['grad'])
        y_norm = jnp.linalg.norm(inner_state['y_k'])
        return jnp.logical_and(
            grad_norm > inner_state['adapt_tol'],  # adjust adapt_tol as needed
            jnp.logical_and(y_norm > 1e-4, inner_state['iter'] < 500)
        )

    def inner_body(inner_state):
        x_curr = inner_state['x']
        grad_curr = inner_state['grad']
        d_curr = inner_state['d']
        alpha = inner_state['alpha']
        mu = inner_state['mu']
        lam = inner_state['lam']
        
        loss_curr = compute_loss(x_curr, JK, H, S, mu, lam)
        x_next = x_curr + alpha * d_curr
        loss_next = compute_loss(x_next, JK, H, S, mu, lam)

        factor = 1e-4 * jnp.dot(grad_curr, d_curr)
        x_next, loss_next, alpha = armijo_line_search(x_curr, x_next, loss_curr, loss_next, d_curr, factor, alpha, JK, H, S, mu, lam)

        grad_next = compute_gradient(x_next, JK, H, S, mu, lam)

        # Secant step calculations
        s_k = x_next - x_curr
        y_k = grad_next - grad_curr
        s_k_sqr = jnp.dot(s_k, s_k)

        # Adaptive rho_k
        theta_k = 6 * (loss_curr - loss_next) + 3 * jnp.dot(grad_curr + grad_next, s_k)
        rho_k = lax.cond(
            s_k_sqr <= 1.0,
            lambda _: 1.0,
            lambda _: 0.0,
            operand=None
        )

        y_k_mod = lax.cond(
            theta_k > 0.0,
            lambda _: y_k + (rho_k * theta_k / s_k_sqr) * s_k,
            lambda _: y_k,
            operand=None
        )

        # Compute beta
        numerator1 = jnp.dot(grad_next, y_k_mod)
        numerator2 = 0.05 * jnp.dot(grad_next, s_k)
        denominator = jnp.dot(d_curr, y_k_mod)

        beta = lax.cond(
            denominator != 0.0,
            lambda _: jnp.maximum(numerator1 / denominator, 0.0) - numerator2 / denominator,
            lambda _: 0.0,
            operand=None
        )

        # Update direction and position
        d_next = -grad_next + beta * d_curr
        alpha = 0.75 * jnp.linalg.norm(s_k) / jnp.linalg.norm(d_curr)        
        
        return {
            'x': x_next,
            'grad': grad_next,
            'd': d_next,
            'alpha': alpha,
            'y_k': y_k,
            'iter': inner_state['iter'] + 1,
            'mu': mu,
            'lam': lam,
            'adapt_tol': inner_state['adapt_tol']
        }

    outer_state = {
        'x': x_curr,
        'constraint': constraint,
        'mu': mu,
        'lam': lam,
        'adapt_tol': adapt_tol,
        'koeff': koeff,
        'iter': 0 
    }
    
    def outer_cond(state):
        return state['constraint'] > 1.0

    def outer_body(state):
        # Initialize inner state
        grad_curr = compute_gradient(state['x'], JK, H, S, state['mu'], state['lam'])
        inner_state = {
            'x': state['x'],
            'grad': grad_curr,
            'd': -grad_curr,
            'alpha': 0.05,
            'y_k': jnp.ones_like(state['x']),
            'iter': state['iter'],
            'mu': state['mu'],
            'lam': state['lam'],
            'adapt_tol': state['adapt_tol']
        }

        final_inner = lax.while_loop(inner_cond, inner_body, inner_state)

        # Update constraint and multipliers
        x_curr = final_inner['x']
        constraint = orthogonality(x_curr, S)
        mu = state['mu'] * 2.0
        lam = state['lam'] + 2.0 * state['mu'] * constraint
        adapt_tol = 0.5 * jnp.exp(-state['koeff'] * mu)

        return {
            'x': final_inner['x'],
            'constraint': constraint,
            'mu': mu,
            'lam': lam,
            'adapt_tol': adapt_tol,
            'koeff': state['koeff'],
            'iter': final_inner['iter']
        }

    final_state = lax.while_loop(outer_cond, outer_body, outer_state)
    return final_state['x'], final_state['constraint'], final_state['adapt_tol'], final_state['mu'], final_state['lam'], final_state['iter']


def run_trust_constr_phase(x_curr, JK, H, S):
    
    def orthogonality_vec(x):
        C = jnp.reshape(x, (ao, occ_o))
        temp = jnp.matmul(S, C)
        temp2 = jnp.matmul(C.T, temp)
        temp3 = temp2 - jnp.eye(occ_o)
        return temp3.flatten()

    def orthogonality_jacobian(x):
        return np.asarray(jax.jacobian(orthogonality_vec)(jnp.array(x)))
    
    def objective(x):
        return float(compute_energy(jnp.array(x), JK, H, S))
    
    def objective_grad(x):
        return np.asarray(jax.grad(compute_energy)(jnp.array(x), JK, H, S))
    
    nonlin_constr = NonlinearConstraint(
        fun=orthogonality_vec,
        lb=0.0, ub=0.0,
        jac=orthogonality_jacobian,
    )
    
    res = minimize(
        fun=objective,
        jac=objective_grad,
        x0=x_curr,
        method='trust-constr',
        constraints=[nonlin_constr],
        #callback=callback,
        hess=BFGS(),
        options={
            'verbose': 0,
            'maxiter': 500,
            'gtol': 1e-4,
            'xtol': 1e-5,
            'barrier_tol': 1e-5,
            #'factorization_method': 'SVDFactorization'
        }
    )
    
    return res.x, res.nit, orthogonality(res.x, S)


def hybrid_optimization(JK, H, S, mu, lam):
    adapt_tol = 1.0
    coeff = 0.001
    tol_grad_change = 1e-4

    start_time = time.time()

    # Initialization                                                                                                                                                        
    key = jax.random.PRNGKey(0)
    x_curr = jax.random.uniform(key, (ao * occ_o,), minval=0.0, maxval=0.1, dtype=jnp.float32)
    constraint = orthogonality(x_curr, S)

    print("Starting Secant Phase...")
    x_curr, constraint, adapt_tol, mu, lam, iteration_secant = run_secant_phase(x_curr, constraint, adapt_tol, coeff, mu, lam, JK, H, S)

    print(f"Lambda = {lam:.4f}, Mu = {mu:.1e}")

    # Trust Regions Phase
    print("Switching to trust regions constrained optimization phase...")
    x_curr, iteration_tr, constraint = run_trust_constr_phase(x_curr, JK, H, S)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimization completed in {iteration_secant} Secant iterations and {iteration_tr} TR iterations.")
    print(f"Final Energy: {compute_energy(x_curr, JK, H, S):.6f}, Constraint: {orthogonality(x_curr, S):.8f}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    return x_curr, iteration_secant + iteration_tr, elapsed_time


# Test the hybrid gradient optimization
times = []
for _ in range(num_repeats):
    mu = 1.0
    lam = 0.0
    optimized_x, iterations, elapsed_time = hybrid_optimization(G, H, S, mu, lam)
    print(f"Final optimized energy: {compute_energy(optimized_x, G, H, S):.4f}")
    times.append(elapsed_time)

average_time = sum(times[1:]) / len(times[1:])
print(f"Average time: {average_time:.4f} seconds")
scf_energy = compute_energy(optimized_x, G, H, S)+nuclear_repulsion_energy
print(f"SCF energy: {scf_energy:.4f}")





