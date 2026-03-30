"""
Cahn-Hilliard PINN Solver — 1D Periodic Domain
================================================
Solves: ∂c/∂t = ∂²/∂x² [ c³ - c - ε²∂²c/∂x² ]
Domain: x ∈ [0,1) periodic, t ∈ [0,1], ε = 0.05

The Cahn-Hilliard equation models phase separation in binary mixtures
(e.g. oil/water, alloy spinodal decomposition). The scalar field c(x,t)
represents the local concentration difference between the two phases:
  c ≈ +1  →  phase A  (e.g. oil-rich region)
  c ≈ -1  →  phase B  (e.g. water-rich region)

A PINN (Physics-Informed Neural Network) replaces a traditional finite-difference
or spectral solver. The network learns c(x,t) by minimising two losses:
  1. PDE residual — how much the network's output violates the equation
  2. IC residual  — how much the network's output at t=0 differs from c(x,0)

Periodicity is enforced EXACTLY via Fourier Feature embedding on x, so
  c(0,t) = c(1,t) for all t by construction — no extra penalty needed.
"""

# --------------------------------------------------------------------------
# Standard library / third-party imports
# --------------------------------------------------------------------------

import torch                              # Core PyTorch tensor library
import torch.nn as nn                     # Neural-network building blocks (Linear, Tanh, etc.)
import numpy as np                        # Numerical arrays — used for IC and grid evaluation
import matplotlib                         # Matplotlib base package
matplotlib.use('Agg')                     # Use non-interactive backend so the script can run
                                          # without a display (e.g. on a server/headless machine)
import matplotlib.pyplot as plt           # Plotting API
from torch.optim.lr_scheduler import CosineAnnealingLR  # Smoothly decays the Adam learning rate

# --------------------------------------------------------------------------
# Global precision setting
# --------------------------------------------------------------------------

# Switch PyTorch's default floating-point type from float32 to float64.
# This is CRITICAL here because we differentiate c four times in sequence
# (∂⁴c/∂x⁴). With float32, accumulated rounding errors in higher-order
# autograd can make the 4th derivative numerically meaningless.
torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------
# Physical and training hyper-parameters (module-level constants)
# --------------------------------------------------------------------------

EPS = 0.05        # ε — interface width parameter in the Cahn-Hilliard equation.
                  # Smaller ε → sharper phase boundaries → harder to train.

N_F = 10_000      # Number of PDE collocation points scattered randomly in (x,t)
                  # space. More points → better PDE coverage → slower per-step.

N_IC = 2_000      # Number of points used to enforce the initial condition c(x,0).
                  # These live on the t=0 line only.

LAMBDA_PDE = 1.0  # Weight (λ_pde) for the PDE loss term in the total loss.
                  # Kept at 1 — the reference scale; other weights are relative to it.

LAMBDA_IC = 100.0 # Weight (λ_ic) for the initial-condition loss term.
                  # Set 100× higher than λ_pde so the network first learns to match
                  # the IC before trying to satisfy the PDE dynamics.
                  # If the solution collapses to c≈0 everywhere, raise this to 1000.

# --------------------------------------------------------------------------
# GPU scaling guide
# --------------------------------------------------------------------------
# float64 throughput by GPU tier (relative to float32):
#   Consumer GPU  (RTX 3090/4090):  1/32 – 1/64   ← serious bottleneck
#   Data-centre   (A100 / H100):    1/2             ← real speedup available
#
# float64 cannot be relaxed — the 4th-order autograd chain needs it.
# On a consumer GPU the CPU and GPU will run at similar speed.
# On an A100/H100 the GPU is genuinely faster; scale up N_F and hidden_dim.
#
# Recommended N_F (PDE collocation points) by hardware:
#   CPU only          →  N_F =  10_000  (default above)
#   Consumer GPU      →  N_F =  20_000  (float64 bottleneck limits benefit)
#   A100 / H100       →  N_F = 100_000  (saturates float64 GEMM units)
#
# Recommended hidden_dim in PINN(...) by hardware:
#   CPU / consumer    →  64   (default)
#   A100 / H100       →  256  (bigger model, better GPU utilisation)
#
# Both N_F and hidden_dim can be changed at the call-site in main()
# without touching any other code.


# ===========================================================================
#  CLASS: FourierEmbedding1D
# ===========================================================================

class FourierEmbedding1D(nn.Module):
    """
    Encodes the spatial coordinate x into a set of sine/cosine features.

    Why do this instead of feeding raw x directly?
    ──────────────────────────────────────────────
    1. EXACT PERIODICITY: sin(2πkx) and cos(2πkx) are both 1-periodic in x,
       so any linear combination of them is also 1-periodic. Because the MLP
       that follows is a function of these features only, the network output
       is guaranteed to satisfy c(0,t) = c(1,t) for every t — no boundary
       loss term needed.

    2. SPECTRAL BIAS FIX: Neural networks with raw (x,t) inputs tend to learn
       low-frequency solutions first (a phenomenon called "spectral bias").
       Fourier features inject explicit high-frequency information so the
       network can represent sharp interfaces.

    Maps (x, t) → [sin(2π·1·x), cos(2π·1·x),
                   sin(2π·2·x), cos(2π·2·x),
                   ...
                   sin(2π·K·x), cos(2π·K·x),
                   t]
    Output shape: (N, 2K + 1)   where the "+1" is for the t channel.
    """

    def __init__(self, K: int = 16):
        """
        Args:
            K: Number of Fourier modes (frequencies 1, 2, …, K).
               K=16 gives 2·16+1 = 33 input features to the MLP.
        """
        super().__init__()   # Initialise the parent nn.Module (sets up parameter tracking)
        self.K = K           # Store K so forward() can reference it

        # Build the integer frequency vector [1, 2, 3, ..., K] as a float64 tensor.
        # Shape: (K,)
        # We'll broadcast this against each incoming x of shape (N,1) to produce (N,K).
        freqs = torch.arange(1, K + 1, dtype=torch.float64)  # → [1., 2., ..., 16.]

        # register_buffer stores the tensor as part of the module's state but
        # does NOT treat it as a trainable parameter. Benefits:
        #   • It automatically moves to the correct device when model.to(device) is called
        #   • It is saved/loaded with model.state_dict()
        #   • Gradients are NOT computed for it
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier feature vector for every (x_i, t_i) pair.

        Args:
            x: Spatial coordinates, shape (N, 1).  Values in [0, 1).
            t: Time coordinates,    shape (N, 1).  Values in [0, 1].

        Returns:
            Tensor of shape (N, 2K+1) containing the embedded features.
        """
        # Multiply each x_i by all K frequencies at once via broadcasting:
        #   x has shape (N, 1)   and   self.freqs has shape (K,)
        #   PyTorch broadcasts them to (N, K) automatically.
        # Result: kx[i, k] = 2π · (k+1) · x[i]     (k is 0-indexed here)
        kx = 2.0 * torch.pi * x * self.freqs  # shape: (N, K)

        # Apply sine and cosine element-wise.
        # Because sin(2πk·0) = sin(2πk·1) = 0 and cos(2πk·0) = cos(2πk·1) = 1
        # for all integer k, both feature matrices are 1-periodic in x.
        sin_feats = torch.sin(kx)  # shape: (N, K)
        cos_feats = torch.cos(kx)  # shape: (N, K)

        # Concatenate [sin features | cos features | t] along the feature axis (dim=1).
        # t is appended last and is NOT embedded — it is passed through unchanged.
        # Time does not need periodicity, so raw t is fine.
        # Final shape: (N, K + K + 1) = (N, 2K+1) = (N, 33) for K=16
        features = torch.cat([sin_feats, cos_feats, t], dim=1)

        return features   # shape (N, 33)


# ===========================================================================
#  CLASS: PINN
# ===========================================================================

class PINN(nn.Module):
    """
    Full Physics-Informed Neural Network.

    Architecture:
        FourierEmbedding1D(x, t)          →  (N, 33)
        Linear(33 → 64)  + Tanh           →  (N, 64)
        Linear(64 → 64)  + Tanh  ×4       →  (N, 64)
        Linear(64 → 1)                    →  (N, 1)   ← predicted c(x,t)

    Why tanh?
    ─────────
    tanh is smooth (infinitely differentiable), which is required because we
    differentiate the network output up to 4th order in x. ReLU would produce
    zero 2nd derivatives almost everywhere; tanh does not have this problem.

    Why Xavier initialisation?
    ──────────────────────────
    Xavier uniform init sets weights so that the variance of activations and
    gradients is preserved across layers, preventing vanishing/exploding
    gradients at the start of training.
    """

    def __init__(self, K: int = 16, hidden_layers: int = 5, hidden_dim: int = 64):
        """
        Args:
            K:             Number of Fourier modes; determines input dimension = 2K+1.
            hidden_layers: Total number of hidden layers (each has hidden_dim neurons).
            hidden_dim:    Width of every hidden layer.
        """
        super().__init__()  # Initialise nn.Module bookkeeping

        # Instantiate the Fourier embedding module.
        # This is a sub-module, so its buffers/parameters are tracked automatically.
        self.embed = FourierEmbedding1D(K)

        # Compute the input dimension to the MLP.
        # The embedding outputs 2K sin/cos features + 1 raw t feature = 2K+1.
        in_dim = 2 * K + 1   # = 33 for K=16

        # Build the MLP as a list of layers, then wrap in nn.Sequential.
        layers = []

        # --- First hidden layer: maps from embedding space to hidden space ---
        layers.append(nn.Linear(in_dim, hidden_dim))  # (33 → 64) weight matrix
        layers.append(nn.Tanh())                       # Non-linearity after the first layer

        # --- Remaining hidden layers: hidden_dim → hidden_dim ---
        # We already added 1 hidden layer above, so we add (hidden_layers - 1) more.
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # (64 → 64) weight matrix
            layers.append(nn.Tanh())                           # Non-linearity

        # --- Output layer: maps hidden representation to scalar c ---
        # No activation here — c is unbounded (though in practice ≈ [-1, +1]).
        layers.append(nn.Linear(hidden_dim, 1))  # (64 → 1)

        # Pack all layers into an nn.Sequential container.
        # Calling self.net(phi) will pass phi through each layer in order.
        self.net = nn.Sequential(*layers)

        # Apply Xavier uniform initialisation to all Linear layers.
        self._init_weights()

    def _init_weights(self):
        """
        Initialise every Linear layer's weights with Xavier uniform and biases to zero.

        Xavier uniform draws weights from U(-a, a) where
            a = gain * sqrt(6 / (fan_in + fan_out))
        gain=1 is the default (appropriate for tanh).
        This keeps the output variance roughly equal to the input variance,
        which helps gradients flow at the start of training.
        """
        for m in self.net:                        # Iterate over every layer in the MLP
            if isinstance(m, nn.Linear):          # Only initialise Linear layers, skip Tanh
                nn.init.xavier_uniform_(m.weight) # Overwrite weight tensor in-place
                nn.init.zeros_(m.bias)            # Start biases at exactly zero

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the network at points (x, t).

        Args:
            x: shape (N, 1) — spatial coordinates
            t: shape (N, 1) — time coordinates

        Returns:
            c: shape (N, 1) — predicted phase-field value at each (x_i, t_i)
        """
        phi = self.embed(x, t)  # Apply Fourier embedding → (N, 2K+1)
        return self.net(phi)    # Pass through MLP → (N, 1)


# ===========================================================================
#  HELPER: _grad
# ===========================================================================

def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Compute the element-wise derivative of `output` with respect to `inp`
    using PyTorch's automatic differentiation engine.

    This is a thin wrapper around torch.autograd.grad that keeps the
    computational graph alive so we can differentiate again afterwards
    (needed for 2nd, 3rd, and 4th derivatives).

    Args:
        output: A tensor of shape (N, 1) that depends on `inp`.
        inp:    A tensor of shape (N, 1) that must have requires_grad=True.

    Returns:
        Tensor of shape (N, 1): ∂(Σ output_i)/∂inp evaluated at each point.

    Why grad_outputs=torch.ones_like(output)?
    ──────────────────────────────────────────
    torch.autograd.grad computes a vector-Jacobian product (VJP).
    We want the simple pointwise derivative, which corresponds to
    multiplying the Jacobian by the all-ones vector.  This effectively
    gives us d(output[i])/d(inp[i]) for each i simultaneously.

    Why create_graph=True?
    ──────────────────────
    This tells autograd to build a graph for the gradient computation itself,
    so we can differentiate the result again (needed for c_xx, c_xxx, c_xxxx).
    Without this, the gradient is a "leaf" tensor and cannot be further differentiated.

    Why retain_graph=True?
    ──────────────────────
    Autograd frees intermediate tensors after computing a gradient to save memory.
    Setting retain_graph=True keeps them alive so we can call _grad again on
    the same forward pass (e.g., to get c_t after already computing c_x).
    """
    return torch.autograd.grad(
        output,                          # The quantity to differentiate
        inp,                             # Differentiate with respect to this
        grad_outputs=torch.ones_like(output),  # VJP vector = [1, 1, ..., 1]
        create_graph=True,               # Keep graph for higher-order derivatives
        retain_graph=True,               # Don't free the graph after this call
    )[0]                                 # [0] because grad() returns a tuple


# ===========================================================================
#  FUNCTION: pde_residual
# ===========================================================================

def pde_residual(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Evaluate the Cahn-Hilliard PDE residual R(x,t) at a batch of collocation points.

    The Cahn-Hilliard equation is:
        ∂c/∂t = ∂²/∂x² [ c³ - c - ε² ∂²c/∂x² ]

    Expanding the right-hand side analytically before coding:
        ∂/∂x [ c³ - c ]  = (3c² - 1) · c_x
        ∂²/∂x² [ c³ - c ] = ∂/∂x [(3c²-1)·c_x]
                           = (3c²-1)·c_xx + 6c·(c_x)²   ← product rule

        ∂²/∂x² [-ε²·c_xx] = -ε²·c_xxxx

    Full residual (rearranging to R = 0):
        R = c_t - (3c²-1)·c_xx - 6c·(c_x)² + ε²·c_xxxx

    Why expand analytically?
    ────────────────────────
    We could write the unexpanded ∂²/∂x²[c³-c] and let autograd compute it,
    but that would require differentiating a product twice inside autograd,
    which creates a longer, more fragile computation graph. The hand-expanded
    form is both more numerically stable and more transparent.

    Args:
        model: The PINN whose output is c(x,t).
        x:     Collocation x-coords, shape (N,1), requires_grad=True.
        t:     Collocation t-coords, shape (N,1), requires_grad=True.
        eps:   Interface width ε (default 0.05).

    Returns:
        R: Residual tensor of shape (N,1). Should be near zero after training.
    """
    # --- Forward pass: evaluate c at all collocation points ---
    c = model(x, t)   # shape (N, 1)

    # --- Time derivative ---
    # c_t[i] = ∂c/∂t evaluated at (x_i, t_i)
    c_t = _grad(c, t)      # shape (N, 1)

    # --- Spatial derivatives (applied sequentially) ---
    # Each call to _grad takes the previous derivative and differentiates again w.r.t. x.

    c_x    = _grad(c,     x)   # First  derivative: ∂c/∂x,     shape (N,1)
    c_xx   = _grad(c_x,   x)   # Second derivative: ∂²c/∂x²,   shape (N,1)
    c_xxx  = _grad(c_xx,  x)   # Third  derivative: ∂³c/∂x³,   shape (N,1)
    c_xxxx = _grad(c_xxx, x)   # Fourth derivative: ∂⁴c/∂x⁴,   shape (N,1)
                                # Required by the bi-harmonic term ε²·∂⁴c/∂x⁴

    # --- Assemble the residual using the analytically expanded PDE ---
    #   Term 1:  c_t                    → ∂c/∂t  (left-hand side)
    #   Term 2: -(3c²-1)·c_xx          → from ∂²/∂x²[c³-c], first part
    #   Term 3: -6c·(c_x)²             → from ∂²/∂x²[c³-c], second part (chain rule)
    #   Term 4: +ε²·c_xxxx             → from -ε²·∂²/∂x²[∂²c/∂x²] = +ε²·∂⁴c/∂x⁴
    R = c_t - (3.0 * c**2 - 1.0) * c_xx - 6.0 * c * c_x**2 + eps**2 * c_xxxx

    return R   # shape (N, 1); L_pde = mean(R²)


# ===========================================================================
#  FUNCTION: ic_function
# ===========================================================================

def ic_function(x, seed: int = 42):
    """
    Compute the exact initial condition c(x, 0).

    The IC is a superposition of three Fourier modes with random amplitudes:
        c(x, 0) = Σ_{k=1}^{3} [ a_k · cos(2πkx) + b_k · sin(2πkx) ]

    where a_k, b_k ~ U(-1, 1) drawn with a fixed seed for reproducibility.

    After summing, the profile is normalised so that max|c(x,0)| = 0.05.
    This small amplitude keeps the initial condition well within the linear
    (early-time) regime of spinodal decomposition, making the problem easier
    to learn while still triggering phase separation at later times.

    Why only k=1,2,3 (no k=0 constant mode)?
    ──────────────────────────────────────────
    ∫₀¹ cos(2πkx) dx = 0  and  ∫₀¹ sin(2πkx) dx = 0  for all integer k ≥ 1.
    Therefore the total mass ∫₀¹ c(x,0) dx = 0 exactly.
    Since Cahn-Hilliard conserves mass, mass(t) = 0 for all t in the true solution,
    making mass drift a clean diagnostic (any drift is a PINN error).

    Args:
        x:    Spatial coordinates. Can be a numpy array of shape (N,) or (N,1),
              or a torch Tensor of shape (N,1).
        seed: Random seed for reproducibility (default 42).

    Returns:
        c values with the same type as x (numpy array or torch Tensor),
        shape (N, 1).
    """
    # Create a numpy random generator with a fixed seed.
    # Using default_rng (Generator API) rather than np.random.seed for
    # better isolation — it does not affect the global numpy random state.
    rng = np.random.default_rng(seed)

    # Draw 3 cosine amplitudes and 3 sine amplitudes from Uniform(-1, 1).
    # a[k-1] is the amplitude for cos(2πkx), b[k-1] for sin(2πkx).
    a = rng.uniform(-1, 1, 3)   # shape (3,): [a_1, a_2, a_3]
    b = rng.uniform(-1, 1, 3)   # shape (3,): [b_1, b_2, b_3]

    if isinstance(x, torch.Tensor):
        # ---- Pure-torch path (GPU-safe) ----
        # All arithmetic stays on the tensor's device (CPU or CUDA).
        # No .cpu() / .numpy() / .tensor() round-trip, so there is no
        # host↔device synchronisation stall — the GPU pipeline is never
        # drained just to compute a simple Fourier sum.

        device = x.device   # remember device so outputs land in the same place

        # Lift the numpy amplitude vectors to float64 tensors on the same device.
        # This is a one-time host→device copy of 6 scalars — negligible cost.
        a_t = torch.tensor(a, dtype=torch.float64, device=device)  # (3,)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)  # (3,)

        # Frequency indices [1, 2, 3] as a device tensor.
        k_t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)  # (3,)

        # Compute all three mode arguments at once via broadcasting:
        #   x:   (N, 1)  ×  k_t: (3,)  →  kx: (N, 3)
        # kx[i, j] = 2π · k_t[j] · x[i]
        kx = 2.0 * torch.pi * x * k_t   # (N, 3)

        # Sum the three cosine and sine contributions along the mode axis (dim=1).
        # a_t·cos(kx) + b_t·sin(kx) has shape (N, 3); sum → (N, 1) with keepdim.
        c = (a_t * torch.cos(kx) + b_t * torch.sin(kx)).sum(dim=1, keepdim=True)  # (N, 1)

        # Normalise so the peak amplitude is exactly 0.05.
        # Guard against the (astronomically unlikely) case where all random
        # amplitudes cancel to zero, which would produce NaN via 0/0.
        den = c.abs().max()
        # .item() extracts a Python float from the 0-dim tensor so that the
        # comparison uses Python's > operator, not tensor.__gt__.  Using a
        # 0-dim tensor directly in an `if` implicitly calls __bool__(), which
        # works but is less explicit and triggers a GPU→CPU sync on CUDA.
        if den.item() > 0:
            c = c / den * 0.05

        return c   # (N, 1) float64 tensor, same device as x

    else:
        # ---- NumPy / list path (used during evaluation on CPU grids) ----
        # Accepts plain numpy arrays or Python lists; returns a numpy array.
        x_np = np.asarray(x).reshape(-1)   # ensure flat 1-D array

        # Initialise the IC profile to zero at every x point.
        c = np.zeros_like(x_np, dtype=np.float64)

        # Accumulate the three Fourier modes.
        # For k=1: c += a[0]·cos(2πx)  + b[0]·sin(2πx)
        # For k=2: c += a[1]·cos(4πx)  + b[1]·sin(4πx)
        # For k=3: c += a[2]·cos(6πx)  + b[2]·sin(6πx)
        for k in range(1, 4):
            c += a[k - 1] * np.cos(2.0 * np.pi * k * x_np)
            c += b[k - 1] * np.sin(2.0 * np.pi * k * x_np)

        # Normalise: divide by max|c| then multiply by 0.05.
        # Guard against the astronomically unlikely zero-profile case.
        den = np.max(np.abs(c))
        if den > 0:
            c = c / den * 0.05

        return c.reshape(-1, 1)   # (N, 1) numpy array


# ===========================================================================
#  FUNCTION: sample_points
# ===========================================================================

def sample_points(N_f: int, N_ic: int, device: str):
    """
    Sample random collocation points for the PDE loss and the IC loss.

    PDE collocation points (x_f, t_f):
        x_f ~ Uniform[0, 1)  — random positions in the spatial domain
        t_f ~ Uniform[0, 1]  — random times throughout the training window
        Both require requires_grad=True so autograd can compute ∂c/∂x and ∂c/∂t.

    IC points (x_ic, t_ic, c_ic):
        x_ic ~ Uniform[0, 1) — random positions on the initial slice
        t_ic = 0              — all live at t=0 by definition
        c_ic = ic_function(x_ic) — precomputed exact IC values (fixed for this batch)
        These do NOT need requires_grad because we only compute c(x_ic, 0)
        and compare it directly to the known IC — no differentiation needed.

    Why precompute c_ic here?
    ─────────────────────────
    ic_function() is deterministic for a given x_ic — the result never
    changes while the batch is held fixed.  Computing it inside
    compute_losses() would repeat an identical calculation on every Adam
    step (10,000×) and on every L-BFGS closure call.  Precomputing once
    here and passing the result in avoids that wasted work entirely.
    (The torch path of ic_function stays on-device via torch.cos/sin,
    so no host↔device transfer is involved.)

    Args:
        N_f:    Number of PDE collocation points.
        N_ic:   Number of initial-condition points.
        device: 'cpu' or 'cuda' — where to allocate tensors.

    Returns:
        x_f:  (N_f,  1) tensor, requires_grad=True
        t_f:  (N_f,  1) tensor, requires_grad=True
        x_ic: (N_ic, 1) tensor
        t_ic: (N_ic, 1) tensor of zeros
        c_ic: (N_ic, 1) tensor — exact IC values c(x_ic, 0), precomputed
    """
    # torch.rand samples from Uniform[0, 1) directly on the target device.
    # requires_grad=True is set directly at creation — this is more efficient
    # than creating the tensor first and then calling .requires_grad_(True).
    x_f = torch.rand(N_f,  1, dtype=torch.float64, device=device, requires_grad=True)
    t_f = torch.rand(N_f,  1, dtype=torch.float64, device=device, requires_grad=True)

    # Sample IC x-coordinates directly on device using torch.rand.
    # Previously this used np.random.uniform + torch.tensor(), which created
    # the array on CPU then copied it to the GPU — an unnecessary host→device
    # transfer on every resample event.  torch.rand(..., device=device) avoids
    # that transfer entirely by allocating and filling the tensor on the GPU.
    # Note: this now draws from the torch RNG (seeded by torch.manual_seed in
    # main()) rather than numpy's RNG, so the exact x_ic values differ from
    # the original numpy-based version — but the distribution is identical.
    x_ic = torch.rand(N_ic, 1, dtype=torch.float64, device=device)

    # All IC points are at time t=0.
    # torch.zeros creates a tensor of all zeros with the same shape convention.
    t_ic = torch.zeros(N_ic, 1, dtype=torch.float64, device=device)

    # Precompute the exact IC values for this batch of x_ic points.
    # ic_function detects the torch tensor and returns a torch tensor on the same device.
    # Computed once here; passed directly to compute_losses() to avoid repeated conversion.
    c_ic = ic_function(x_ic)   # shape (N_ic, 1)

    return x_f, t_f, x_ic, t_ic, c_ic


# ===========================================================================
#  FUNCTION: compute_losses
# ===========================================================================

def compute_losses(model, x_f, t_f, x_ic, t_ic, c_exact_ic):
    """
    Compute the three loss components from the current network state.

    L_pde   = mean(R²)                       — PDE residual squared, averaged
    L_ic    = mean((c_pred - c_exact)²)      — IC mismatch squared, averaged
    L_total = λ_pde · L_pde + λ_ic · L_ic   — weighted sum

    Args:
        model:       The PINN to evaluate.
        x_f:         PDE collocation x-coords, (N_f, 1), requires_grad=True.
        t_f:         PDE collocation t-coords, (N_f, 1), requires_grad=True.
        x_ic:        IC x-coords, (N_ic, 1).
        t_ic:        IC t-coords (all zero), (N_ic, 1).
        c_exact_ic:  Precomputed exact IC values c(x_ic, 0), (N_ic, 1).
                     Passed in from sample_points() so it is computed once
                     per batch rather than on every step or closure call.

    Returns:
        (L_total, L_pde, L_ic) — all scalar tensors, attached to the graph.
    """
    # --- PDE loss ---
    # Compute the residual R at every collocation point.
    # R should be zero if the network perfectly satisfies the Cahn-Hilliard equation.
    R = pde_residual(model, x_f, t_f)      # shape (N_f, 1)

    # Mean squared residual: L_pde = (1/N_f) Σ R_i²
    # Squaring makes the loss non-negative and penalises large violations more strongly.
    L_pde = torch.mean(R**2)               # scalar

    # --- IC loss ---
    # Ask the network what it predicts at the initial time for the IC x-locations.
    c_pred_ic = model(x_ic, t_ic)          # shape (N_ic, 1)  — network prediction at t=0

    # c_exact_ic is the precomputed ground-truth IC, passed in from sample_points().
    # Mean squared error between prediction and exact IC.
    L_ic = torch.mean((c_pred_ic - c_exact_ic)**2)  # scalar

    # --- Weighted total loss ---
    # λ_ic = 100 forces the network to strongly honour the IC before worrying
    # about PDE accuracy (PDE loss is typically large early in training).
    L_total = LAMBDA_PDE * L_pde + LAMBDA_IC * L_ic  # scalar

    return L_total, L_pde, L_ic


# ===========================================================================
#  FUNCTION: train
# ===========================================================================

def train(model: PINN, device: str):
    """
    Train the PINN in two sequential phases:

    Phase 1 — Adam (stochastic first-order optimiser):
        Fast, noisy descent. Gets the network into a good basin quickly.
        10,000 steps with cosine-annealed learning rate 1e-3 → 1e-5.
        Collocation points are resampled every 1,000 steps so the network
        sees diverse (x,t) pairs and does not overfit a fixed grid.

    Phase 2 — L-BFGS (quasi-Newton second-order optimiser):
        Slow, accurate descent. Refines the solution to machine precision
        using curvature information (approximate Hessian).
        Uses a fixed full batch (no resampling) for stable convergence.

    Args:
        model:  The PINN to train (modified in-place).
        device: 'cpu' or 'cuda'.

    Returns:
        history: dict with keys 'step', 'total', 'pde', 'ic' — loss values
                 recorded every 500 steps, for plotting.
    """
    # Initialise empty history lists.
    # We append to these every 500 steps during both phases.
    history = {'step': [], 'total': [], 'pde': [], 'ic': []}

    # -----------------------------------------------------------------------
    # Phase 1: Adam
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Adam optimiser (4,000 steps)")
    print("=" * 60)

    # Adam (Adaptive Moment Estimation) maintains per-parameter running estimates
    # of the first and second moments of the gradient, enabling an adaptive
    # effective learning rate per parameter.
    # lr=1e-3 is a standard starting point for PINNs.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Cosine annealing smoothly reduces the learning rate from lr (1e-3) to
    # eta_min (1e-5) following a cosine curve over T_max steps.
    # This avoids oscillating around the minimum at the end of Adam training.
    scheduler = CosineAnnealingLR(optimizer, T_max=10_000, eta_min=1e-5)

    for step in range(1, 10_001):   # steps 1 through 10,000 inclusive

        # Re-draw collocation points every 1,000 steps.
        # This is a form of Monte Carlo integration over (x,t) space —
        # periodically refreshing the points prevents the network from
        # memorising a specific fixed point set (reduces overfitting to grid).
        # Condition: step % 1000 == 1 triggers at steps 1, 1001, 2001, ...
        # The first sample happens here at step=1, so no pre-loop sample is needed.
        if step % 1_000 == 1:
            x_f, t_f, x_ic, t_ic, c_ic = sample_points(N_F, N_IC, device)

        # Zero all accumulated gradients from the previous step.
        # PyTorch accumulates gradients by default; we must reset each step.
        optimizer.zero_grad()

        # Forward pass + loss computation.
        # c_ic (precomputed exact IC) is passed in to avoid a numpy round-trip every step.
        L_total, L_pde, L_ic = compute_losses(model, x_f, t_f, x_ic, t_ic, c_ic)

        # Backward pass: compute ∂L_total/∂θ for every network parameter θ
        # via automatic differentiation (backpropagation through all operations).
        L_total.backward()

        # Gradient clipping: rescale the gradient vector if its L2 norm > 1.0.
        # Prevents large gradient spikes (common with 4th-order PDE losses)
        # from causing unstable weight updates early in training.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Adam parameter update: θ ← θ - lr · m̂ / (√v̂ + ε)
        optimizer.step()

        # Advance the cosine learning-rate schedule by one step.
        scheduler.step()

        # Log progress every 500 steps.
        if step % 500 == 0:
            # .item() extracts a Python float from the scalar tensor,
            # detaching it from the computation graph.
            lt = L_total.item()
            lp = L_pde.item()
            li = L_ic.item()

            # Append to history for later plotting.
            history['step'].append(step)
            history['total'].append(lt)
            history['pde'].append(lp)
            history['ic'].append(li)

            # Print a one-line summary to stdout.
            # :6d pads step to 6 characters; .3e uses scientific notation.
            print(f"  Step {step:6d} | L_total={lt:.3e} | L_pde={lp:.3e} | L_ic={li:.3e}")

    # -----------------------------------------------------------------------
    # Phase 2: L-BFGS
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 2: L-BFGS optimiser")
    print("=" * 60)

    # Release any GPU memory that the Adam phase's retained autograd graphs
    # were holding.  After the last Adam step's .backward() call the graphs
    # are freed automatically, but CUDA's allocator keeps the memory in a
    # pool.  empty_cache() returns it to the OS / driver, which can reduce
    # fragmentation and give L-BFGS (which retains 100 past gradient vectors)
    # a clean pool to allocate from.  On CPU this is a no-op.
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Draw a fresh, fixed batch for L-BFGS.
    # L-BFGS requires a CONSISTENT loss function across the closure calls
    # within a single step (it evaluates the closure multiple times to do
    # line search). Resampling inside the closure would break this consistency,
    # so we fix the batch here and never change it.
    x_f, t_f, x_ic, t_ic, c_ic = sample_points(N_F, N_IC, device)

    # Instantiate the L-BFGS optimiser.
    #   max_iter:        Maximum number of optimisation iterations.
    #   tolerance_grad:  Stop when gradient norm < 1e-9.
    #   tolerance_change: Stop when change in loss < 1e-12 (extremely tight).
    #   history_size:    Number of past gradient/step pairs kept for the
    #                    Hessian approximation. 100 gives very accurate curvature.
    #   line_search_fn:  'strong_wolfe' ensures the Armijo and curvature
    #                    conditions are satisfied, giving robust convergence.
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=20_000,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn='strong_wolfe',
    )

    # L-BFGS in PyTorch REQUIRES a closure function.
    # The closure must:
    #   1. Zero the gradients
    #   2. Re-compute the loss (forward pass)
    #   3. Call .backward() on the loss
    #   4. Return the scalar loss value
    # L-BFGS calls the closure multiple times per step (for line search).

    # Use a mutable list [0] as a counter because Python closures cannot
    # rebind a bare integer variable from the enclosing scope (only read it).
    lbfgs_step = [0]

    def closure():
        # Step 1: zero gradients
        optimizer_lbfgs.zero_grad()

        # Step 2: forward pass — recompute all losses from the fixed batch
        L_total, L_pde, L_ic = compute_losses(model, x_f, t_f, x_ic, t_ic, c_ic)

        # Step 3: backward pass — compute gradients w.r.t. all parameters
        L_total.backward()

        # Increment the iteration counter stored in the mutable list.
        lbfgs_step[0] += 1

        # Log every 500 closure calls.
        # NOTE: lbfgs_step[0] counts CLOSURE CALLS, not true L-BFGS iterations.
        # L-BFGS calls the closure multiple times per iteration during line search,
        # so the closure count is typically 2–4× the iteration count.
        if lbfgs_step[0] % 500 == 0:
            lt = L_total.item()
            lp = L_pde.item()
            li = L_ic.item()

            # Offset step count by 10,000 so history is contiguous with Phase 1.
            history['step'].append(10_000 + lbfgs_step[0])
            history['total'].append(lt)
            history['pde'].append(lp)
            history['ic'].append(li)

            print(f"  L-BFGS closure {lbfgs_step[0]:6d} | L_total={lt:.3e} | "
                  f"L_pde={lp:.3e} | L_ic={li:.3e}")

        # Step 4: return the loss so L-BFGS can use it for line-search decisions.
        return L_total

    # Run the optimiser. Internally it calls closure() many times,
    # each time evaluating the loss, computing gradients, and deciding
    # on a step direction and step length using the strong Wolfe conditions.
    optimizer_lbfgs.step(closure)

    # --- Final loss evaluation ---
    # NOTE: torch.no_grad() cannot be used here. pde_residual() calls
    # torch.autograd.grad() internally, which requires a computation graph
    # to exist. Under no_grad(), the forward pass builds no graph (c has no
    # grad_fn), so autograd.grad() would raise a RuntimeError.
    # We accept the small overhead of a full forward+grad pass just for this
    # one diagnostic print.
    L_total, L_pde, L_ic = compute_losses(model, x_f, t_f, x_ic, t_ic, c_ic)
    print(f"\nFinal | L_total={L_total.item():.3e} | "
          f"L_pde={L_pde.item():.3e} | L_ic={L_ic.item():.3e}")

    return history   # Return the recorded loss values for plotting


# ===========================================================================
#  FUNCTION: evaluate_and_plot  (standalone, no loss history)
# ===========================================================================

def evaluate_and_plot(model: PINN, device: str):
    """
    Evaluate the trained model on a uniform 512×200 grid and produce
    a 2×2 figure. This version omits loss curves (no history available).
    Saves the figure to 'cahn_hilliard_results.png'.

    Returns:
        C:      (512, 200) numpy array of predicted c values.
        t_grid: (200,) time coordinate array.
        x_grid: (512,) space coordinate array.
    """
    N_x, N_t = 512, 200   # Grid resolution: 512 spatial × 200 temporal points

    # Spatial grid uses endpoint=False because the domain is [0, 1) periodic.
    # x = 1.0 is the same physical point as x = 0.0; including it would
    # duplicate the endpoint, skewing snapshot plots, the heatmap, and the
    # trapezoidal mass integral (which would double-count one point).
    # Time is NOT periodic, so t_grid keeps endpoint=True (default).
    x_grid = np.linspace(0, 1, N_x, endpoint=False)  # shape (512,): x = 0, 1/512, ..., 511/512
    t_grid = np.linspace(0, 1, N_t)                   # shape (200,): t = 0, 1/199, ..., 1

    # Build a 2-D meshgrid with 'ij' indexing so that:
    #   XX[i, j] = x_grid[i]  (x varies along axis 0)
    #   TT[i, j] = t_grid[j]  (t varies along axis 1)
    # Both have shape (N_x, N_t) = (512, 200).
    XX, TT = np.meshgrid(x_grid, t_grid, indexing='ij')

    # Flatten the 2-D grids to 1-D column vectors so we can pass them
    # through the network in one batched call.
    # shape: (512*200, 1) = (102400, 1)
    x_flat = XX.reshape(-1, 1)
    t_flat = TT.reshape(-1, 1)

    # Convert numpy arrays to torch tensors on the chosen device.
    # No requires_grad needed — we're doing inference, not computing gradients.
    x_ten = torch.tensor(x_flat, dtype=torch.float64, device=device)
    t_ten = torch.tensor(t_flat, dtype=torch.float64, device=device)

    # Switch model to evaluation mode.
    # For this architecture there are no Dropout or BatchNorm layers, so
    # eval() has no practical effect, but it is good practice.
    model.eval()

    with torch.no_grad():   # Disable autograd to save memory and speed up inference
        chunk = 50_000      # Process 50,000 points at a time to avoid OOM on large GPUs
        c_parts = []        # Accumulate outputs from each chunk

        # Loop over the flattened grid in strides of `chunk`.
        for i in range(0, len(x_ten), chunk):
            # Slice out a chunk of points, run the forward pass, move result to CPU,
            # convert to numpy. .cpu() is a no-op if already on CPU.
            c_parts.append(
                model(x_ten[i:i+chunk], t_ten[i:i+chunk]).cpu().numpy()
            )

        # Concatenate all chunks back into a single array of shape (102400, 1).
        c_flat = np.concatenate(c_parts, axis=0)

    # Reshape back to the 2-D spatial-temporal grid.
    C = c_flat.reshape(N_x, N_t)   # shape (512, 200)

    # ---- Mass conservation diagnostics ----
    print()
    print("=" * 60)
    print("Mass Conservation Check")
    print("=" * 60)

    # Five snapshot times at which we will check the total mass.
    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    masses = []

    for t_star in snap_times:
        # Find the column index in C corresponding to the closest time in t_grid.
        idx = np.argmin(np.abs(t_grid - t_star))   # scalar integer index

        # Extract the spatial profile at that time: shape (512,).
        c_snap = C[:, idx]

        # Integrate using the trapezoidal rule: mass = ∫₀¹ c(x,t*) dx.
        # On a uniform grid of N points spanning [0,1], trapz gives an
        # accurate approximation with O(Δx²) error.
        mass = np.trapz(c_snap, x_grid)   # scalar
        masses.append(mass)
        print(f"  t={t_star:.2f} | mass = {mass:.6f}")

    print()
    for i, t_star in enumerate(snap_times):
        # Report the absolute drift from the initial mass.
        drift = abs(masses[i] - masses[0])
        print(f"  |mass(t={t_star:.2f}) - mass(0)| = {drift:.6e}")

    # ---- Plotting ----
    # Create a 2-row × 2-column grid of subplots.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cahn–Hilliard PINN Solution (1D Periodic)", fontsize=14)

    # --- Panel 1 (top-left): Space-time heatmap ---
    ax = axes[0, 0]
    im = ax.imshow(
        C,                  # 2-D array to display: rows = x, columns = t
        aspect='auto',      # Don't force square pixels; fill the axes box
        origin='lower',     # Place (x=0, t=0) at the bottom-left corner
        extent=[0, 1, 0, 1],# Map pixel coordinates to actual (t, x) ranges
        cmap='RdBu_r',      # Red = positive (phase A), Blue = negative (phase B)
        vmin=-1, vmax=1,    # Fix colour scale to the physically meaningful range
    )
    ax.set_xlabel('t')                                 # Horizontal axis = time
    ax.set_ylabel('x')                                 # Vertical axis = space
    ax.set_title('Space–time heatmap c(x,t)')
    fig.colorbar(im, ax=ax, label='c')                 # Add colour bar

    # --- Panel 2 (top-right): Snapshot curves at five times ---
    ax = axes[0, 1]
    # Generate 5 evenly spaced colours from the viridis colourmap.
    colors = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))

    for t_star, col in zip(snap_times, colors):
        idx = np.argmin(np.abs(t_grid - t_star))   # Find nearest time index
        ax.plot(x_grid, C[:, idx], color=col, label=f't={t_star:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('c')
    ax.set_title('Snapshots c(x, t*)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)   # Restrict x-axis to the physical domain [0, 1]

    # --- Panel 3 (bottom-left): Loss curves placeholder ---
    ax = axes[1, 0]
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss curves (log scale)')
    ax.set_yscale('log')   # Losses span many orders of magnitude; log scale is clearer
    ax.text(0.5, 0.5, 'No loss history\n(pass history to plot)',
            ha='center', va='center', transform=ax.transAxes, color='gray')

    # --- Panel 4 (bottom-right): Mass conservation over time ---
    ax = axes[1, 1]
    # Compute the total mass at every one of the 200 time steps.
    mass_all = np.array([np.trapz(C[:, i], x_grid) for i in range(N_t)])

    ax.plot(t_grid, mass_all, 'b-', linewidth=1.5)   # Blue line = actual mass(t)
    ax.axhline(                                        # Red dashed line = initial mass
        masses[0], color='r', linestyle='--', linewidth=1, label='mass(0)'
    )
    ax.set_xlabel('t')
    ax.set_ylabel('∫c dx')
    ax.set_title('Mass conservation')
    ax.legend()

    plt.tight_layout()   # Adjust subplot spacing to prevent overlap

    fname = 'cahn_hilliard_results.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')   # Save at 150 DPI, no whitespace clipping
    plt.close(fig)   # Release the figure from memory
    print(f"\nPlots saved to {fname}  (no loss curves — run with history)")

    return C, t_grid, x_grid


# ===========================================================================
#  FUNCTION: plot_with_history  (full 4-panel plot including loss curves)
# ===========================================================================

def plot_with_history(model: PINN, device: str, history: dict):
    """
    Evaluate the trained model on a 512×200 grid and produce the full
    4-panel diagnostic figure, including loss curves from training.

    Args:
        model:   Trained PINN.
        device:  'cpu' or 'cuda'.
        history: Dict returned by train() with keys 'step','total','pde','ic'.
    """
    N_x, N_t = 512, 200   # Evaluation grid size

    # Spatial grid uses endpoint=False — domain is [0, 1) periodic, so x = 1
    # is the same point as x = 0. Including it would duplicate the endpoint
    # in plots and double-count one point in the trapezoidal mass integral.
    # Time is not periodic so t_grid keeps the default endpoint=True.
    x_grid = np.linspace(0, 1, N_x, endpoint=False)  # 512 points: 0, 1/512, ..., 511/512
    t_grid = np.linspace(0, 1, N_t)                   # 200 snapshots: 0, ..., 1

    # Construct a 2-D meshgrid (ij indexing: first index = x, second = t).
    XX, TT = np.meshgrid(x_grid, t_grid, indexing='ij')   # both (512, 200)

    # Flatten to column vectors for batch network evaluation.
    x_flat = XX.reshape(-1, 1)   # (102400, 1)
    t_flat = TT.reshape(-1, 1)   # (102400, 1)

    # Move to torch tensors on the target device.
    x_ten = torch.tensor(x_flat, dtype=torch.float64, device=device)
    t_ten = torch.tensor(t_flat, dtype=torch.float64, device=device)

    model.eval()   # Set to inference mode

    with torch.no_grad():   # No gradient tracking needed during evaluation
        chunk = 50_000      # Chunk size to avoid GPU memory overflow
        c_parts = []

        for i in range(0, len(x_ten), chunk):
            c_parts.append(
                model(x_ten[i:i+chunk], t_ten[i:i+chunk]).cpu().numpy()
            )

        # Reassemble all chunks into (102400, 1).
        c_flat = np.concatenate(c_parts, axis=0)

    # Reshape to (512, 200): rows = x positions, columns = time steps.
    C = c_flat.reshape(N_x, N_t)

    # ---- Mass conservation ----
    print()
    print("=" * 60)
    print("Mass Conservation Check")
    print("=" * 60)

    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]   # Diagnostic snapshot times
    masses = []

    for t_star in snap_times:
        idx = np.argmin(np.abs(t_grid - t_star))  # Index of nearest t in t_grid
        mass = np.trapz(C[:, idx], x_grid)         # ∫c dx via trapezoidal rule
        masses.append(mass)
        print(f"  t={t_star:.2f} | mass = {mass:.6f}")

    print()
    for i, t_star in enumerate(snap_times):
        # Absolute deviation of mass at time t_star from mass at t=0.
        print(f"  |mass(t={t_star:.2f}) - mass(0)| = {abs(masses[i] - masses[0]):.6e}")

    # ---- Build 2×2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cahn–Hilliard PINN Solution (1D Periodic)", fontsize=14)

    # --- Panel 1: Space-time heatmap ---
    ax = axes[0, 0]
    im = ax.imshow(
        C,
        aspect='auto',       # Let matplotlib choose aspect ratio
        origin='lower',      # x=0 at bottom
        extent=[0, 1, 0, 1], # Axis labels correspond to actual (t, x) values
        cmap='RdBu_r',       # Diverging colourmap centred at zero
        vmin=-1, vmax=1,     # Full physical range of c
    )
    ax.set_xlabel('t'); ax.set_ylabel('x')
    ax.set_title('Space–time heatmap c(x,t)')
    fig.colorbar(im, ax=ax, label='c')   # Colour bar showing c scale

    # --- Panel 2: Snapshot lines at 5 times ---
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))  # Distinct colours per time

    for t_star, col in zip(snap_times, colors):
        idx = np.argmin(np.abs(t_grid - t_star))         # Nearest time index
        ax.plot(x_grid, C[:, idx], color=col, label=f't={t_star:.2f}')

    ax.set_xlabel('x'); ax.set_ylabel('c')
    ax.set_title('Snapshots c(x, t*)')
    ax.legend(fontsize=8); ax.set_xlim(0, 1)

    # --- Panel 3: Loss curves (log scale) ---
    ax = axes[1, 0]
    # history['step'] mixes two different quantities:
    #   x ≤ 4,000  →  Adam optimiser step number   (1 unit = 1 parameter update)
    #   x > 4,000  →  4,000 + L-BFGS closure call (1 unit = 1 closure invocation,
    #                  not 1 L-BFGS iteration; line search calls closure 2–4× per iter)
    steps = history['step']

    # Plot each loss component on a log y-axis.
    ax.semilogy(steps, history['total'], 'k-',  linewidth=1.5, label='L_total')  # black solid
    ax.semilogy(steps, history['pde'],   'b--', linewidth=1.2, label='L_pde')    # blue dashed
    ax.semilogy(steps, history['ic'],    'r:',  linewidth=1.2, label='L_ic')     # red dotted

    # Vertical line separates the two phases.  x-axis units change at this boundary.
    ax.axvline(10_000, color='gray', linestyle=':', linewidth=1, label='Adam→L-BFGS')

    # Label explicitly reflects that the axis is not a uniform "step" count.
    ax.set_xlabel('Adam step  |  L-BFGS closure call (offset +10 000)'); ax.set_ylabel('Loss')
    ax.set_title('Loss curves (log scale)')
    ax.legend(fontsize=8)

    # --- Panel 4: Mass conservation curve ---
    ax = axes[1, 1]

    # Compute the total integrated mass at every time step.
    # list comprehension: for each column i (time step), integrate c over x.
    mass_all = np.array([np.trapz(C[:, i], x_grid) for i in range(N_t)])

    ax.plot(t_grid, mass_all, 'b-', linewidth=1.5)  # Actual mass vs time

    # Reference line at the initial mass value.
    # A perfectly conservative solver would have mass_all = masses[0] everywhere.
    ax.axhline(masses[0], color='r', linestyle='--', linewidth=1, label='mass(0)')

    ax.set_xlabel('t'); ax.set_ylabel('∫c dx')
    ax.set_title('Mass conservation')
    ax.legend()

    plt.tight_layout()   # Prevent subplot label overlap

    fname = 'cahn_hilliard_results.png'
    # dpi=150 gives a high-resolution image; bbox_inches='tight' removes excess whitespace.
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)   # Free figure memory immediately after saving
    print(f"\nFigure saved → {fname}")


# ===========================================================================
#  FUNCTION: main
# ===========================================================================

def main():
    """
    Entry point: set seeds, create the model, run training, evaluate and plot.
    """
    # Fix the PyTorch random seed so that weight initialisation and all
    # torch.rand calls in sample_points (x_f, t_f, x_ic) are reproducible.
    torch.manual_seed(42)
    # Note: np.random.seed() is intentionally omitted.
    # ic_function() uses np.random.default_rng(seed=42) — its own isolated
    # RNG seeded at call time, unaffected by the global numpy state.
    # x_ic is now drawn via torch.rand, so numpy's global RNG is not used
    # anywhere in the training path.

    # Choose the compute device.
    # If a CUDA-capable GPU is available, use it (much faster for large batches).
    # Fall back to CPU if not. The model and all tensors are created on this device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print a configuration summary for reference.
    print(f"Device: {device}")
    print(f"PyTorch dtype: float64")
    print(f"ε = {EPS},  λ_pde = {LAMBDA_PDE},  λ_ic = {LAMBDA_IC}")
    print(f"N_f = {N_F},  N_ic = {N_IC}\n")

    # Instantiate the PINN and move all parameters and buffers to the chosen device.
    model = PINN(K=16, hidden_layers=5, hidden_dim=64).to(device)

    # Count and display the total number of trainable parameters.
    # sum(...) iterates over all parameter tensors; .numel() returns element count.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # Phase 1 (Adam) + Phase 2 (L-BFGS).
    # Returns the recorded loss values at 500-step intervals.
    history = train(model, device)

    # Evaluate on a dense grid, compute mass conservation, and save the figure.
    plot_with_history(model, device, history)

    # Persist the trained weights to disk.
    # state_dict() captures all learnable parameters and registered buffers.
    # Load later with: model.load_state_dict(torch.load('cahn_hilliard_pinn.pt'))
    torch.save(model.state_dict(), 'cahn_hilliard_pinn.pt')
    print("Model saved → cahn_hilliard_pinn.pt")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

# Only run main() when this file is executed directly (python cahn_hilliard_pinn.py),
# not when it is imported as a module by another script.
if __name__ == '__main__':
    main()
