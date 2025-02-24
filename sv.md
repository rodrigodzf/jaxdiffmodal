# Stormer-verlet integrators


The stormer-verlet is a centered difference scheme used to approximate
derivatives. In the present case we use it for approximating the second
and first derivatives of an oscillator defined by the following
differential equation:

$$
\ddot{q} + c \dot{q} + k q = f(t)
$$

The finite difference operators are defined as:

$$
\begin{align}
\delta_t q &= \frac{q^{n+1} - q^{n-1}}{2 h} \\
\delta\_{tt} q &= \frac{q^{n+1} - 2 q^n + q^{n-1}}{h^2}
\end{align}
$$

for the first derivative and the second derivative respectively. The
difference equation for the oscillator is then:

*δ*<sub>*t**t*</sub>*q* + *c**δ*<sub>*t*</sub>*q* + *k**q* = *f*(*t*)

after expanding and some algebraic manipulation to isolate
*q*<sup>*n* + 1</sup> we get:

$$
\bigl(\tfrac{1}{h^2} + \tfrac{c}{2h}\bigr)\\q^{n+1} +
\bigl(-\tfrac{2}{h^2} + k\bigr) q^n +
\bigl(\tfrac{1}{h^2} - \tfrac{c}{2h}\bigr)\\q^{n-1} =
f(t^n).
$$

$$
\boxed{
q^{n+1} = \frac{2 h^2}{2 + c h}
\Bigl\[
f(t^n) + \Bigl(\tfrac{2}{h^2} - k\Bigr) q^n +
\Bigl(-\tfrac{1}{h^2} + \tfrac{c}{2h}\Bigr) q^{n-1}
\Bigr\].
}
$$

To make it more readable we can define the following constants:

$$
\begin{align}
a &= \frac{2 h^2}{2 + c h} \\
b &= \frac{2}{h^2} - k \\
c &= -\frac{1}{h^2} + \frac{c}{2h}
\end{align}
$$

so that the equation becomes:

$$
\boxed{
q^{n+1} = a \Bigl\[ f(t^n) + b q^n + c q^{n-1} \Bigr\].
}
$$

Since we have *h*<sup>2</sup> in the denominator we can multiply the
whole equation by *h*<sup>2</sup> to get rid of it. Then the constants
become:

$$
\begin{align}
a &= \frac{2}{2 + c h} \\
b &= 2 - k h^2 \\
c &= -1 + \frac{c h}{2}
\end{align}
$$

Note that we also need to multiply the forcing term and the eventual
non-linear terms by *h*<sup>2</sup>. Since we also divide all *k* and
*c* by *ρ* we can define a new constant $g = \frac{h^2}{\rho}$ so that
the equation becomes:

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

``` python
import einops
import jax
import jax.numpy as jnp
```

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

``` python
def A_inv_vector(
    h,  # temporal grid spacing (scalar)
    damping,  # damping term (vector)
):
    """
    Also this is already multiplied $h^2$
    """
    return 2.0 * h**2 / (2.0 + damping * h)
    # return 2.0 / (2.0 + damping * h)


def B_vector(
    h,  # temporal grid spacing (scalar)
    stiffness,  # stiffness term (vector)
):
    """
    Note this include the minus side caused by puting this term on the right hand side of the equation. Also this is already multiplied $h^2$
    """
    return 2.0 / h**2 - stiffness

    # return 2.0 - stiffness * h**2


def C_vector(
    h,  # temporal grid spacing (scalar)
    damping,  # damping term (vector)
):
    """
    Note this include the minus side caused by puting this term on the right hand side of the equation. Also this is already multiplied $h^2$
    """
    return -1.0 / h**2 + (damping / (2.0 * h))
    # return -1.0 + damping * h * 0.5
```

:::

### Stormer-verlet for the tension modulated case

$$
D \Delta \Delta w + \rho \ddot{w} + \left(d_1 + d_3 \Delta\right)\dot{w}- T_0 \Delta w = f\_{ext} - T\_{nl} \Delta w
$$

we can rearrange the terms to isolate have it in the form:

$$
\rho \ddot{w} + \left(d_1 + d_3 \Delta\right)\dot{w} + (D \Delta \Delta - T_0 \Delta) w = f\_{ext} - T\_{nl} \Delta w
$$

By applying the SLT transformation to get rid of the spatial derivatives
we get:

$$
\rho \ddot{q} + \left(d_1 + d_3 \lambda \right)\dot{q} + (D \lambda^2 - T_0 \lambda) q = f\_{ext} - \bar{T}\_{nl} \lambda q
$$

now by applying the previously defined differece operators we get:

$$ q^{n+1} = a .

$$

where the constants are:

$$
\begin{align}
a &= g\left(\frac{2}{2 + d_1 + d_3 \lambda h}\right) \\
b &= g\left(2 -  D \lambda^2 - T_0 \lambda h^2\right) \\
c &= g\left(-1 +  \frac{d_1 + d_3 \lambda h}{2}\right)
\end{align}
$$

we know that
$\bar{T}\_{N L}(q)=\frac{1}{2} \frac{E A}{L} \sum\_\eta \frac{\lambda\_\eta q\_\eta^2(t)}{\left\\K\_\eta\right\\\_2^2}$
from Avanzini and Trautmann.

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

``` python
@jax.jit
def solve_sv_berger_jax_scan(
    A_inv: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    modal_excitation: jnp.ndarray,  # (T, n_modes)
    lambda_mu: jnp.ndarray,
    factors: jnp.ndarray,
    g: float,  # factor for the input
):
    n_modes = A_inv.shape[0]
    q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
    q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

    def advance_state(
        state: jnp.ndarray,  # inital state
        x: jnp.ndarray,  # input
    ) -> tuple[jnp.ndarray, jnp.ndarray]:  # carry, output
        # unpack state
        q_prev, q = state

        nl = lambda_mu * q * (factors @ q**2)

        # compute the next state
        # q_next = B * q + C * q_prev - g * nl + g * x
        q_next = B * q + C * q_prev - g * nl + x

        # return the next state and the output
        return (q, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q_prev, q),
        modal_excitation,  # (T, n_modes)
        unroll=8,
    )
    return state, final
```

:::

``` python
# # | export


# @jax.jit
# def solve_sv_vk_jax_scan(
#     A_inv: jnp.ndarray,
#     B: jnp.ndarray,
#     C: jnp.ndarray,
#     modal_excitation: jnp.ndarray,  # (T, n_modes)
#     Hv: jnp.ndarray,
#     zetafourth: jnp.ndarray,
#     C_NL: float,
#     g: float,  # factor for the input
# ):
#     n_modes = A_inv.shape[0]
#     q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
#     q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

#     # precompute Hv divided by the eigenvalues to make it a bit faster
#     Hva = Hv / (2.0 * zetafourth)

#     def advance_state(
#         state: jnp.ndarray,  # inital state
#         x: jnp.ndarray,  # input
#     ) -> tuple[jnp.ndarray, jnp.ndarray]:  # carry, output
#         # unpack state
#         q_prev, q = state

#         nl = C_NL * einops.einsum(
#             Hva,
#             Hv,
#             q,
#             q,
#             q,
#             "m i j, k l j, i, k, l -> m",
#         )
#         # compute the next state
#         q_next = A_inv * (B * q + C * q_prev - g * nl + g * x)

#         # return the next state and the output
#         return (q, q_next), q_next

#     state, final = jax.lax.scan(
#         advance_state,
#         (q_prev, q),
#         modal_excitation,  # (T, n_modes)
#         unroll=8,
#     )
#     return state, final
```

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

``` python
@jax.jit
def solve_sv_vk_jax_scan(
    A_inv: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    modal_excitation: jnp.ndarray,  # (T, n_modes)
    Hv: jnp.ndarray,
    g: float,  # factor for the input
):
    n_modes = A_inv.shape[0]
    q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
    q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

    def advance_state(
        state: jnp.ndarray,  # inital state
        x: jnp.ndarray,  # input
    ) -> tuple[jnp.ndarray, jnp.ndarray]:  # carry, output
        # unpack state
        q_prev, q = state

        nl = einops.einsum(
            Hv,
            Hv,
            q,
            q,
            q,
            "n p q, n r s, p, q, r -> s",
        )
        # compute the next state
        q_next = B * q + C * q_prev - g * nl + x

        # return the next state and the output
        return (q, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q_prev, q),
        modal_excitation,  # (T, n_modes)
        unroll=8,
    )
    return state, final
```

:::
