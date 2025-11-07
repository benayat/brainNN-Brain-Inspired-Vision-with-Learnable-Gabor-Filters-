# Gabor Filter Algorithm - Deep Dive

## Core Mathematical Foundation

### What is a Gabor Filter?

A **Gabor filter** is the product of a **Gaussian envelope** and a **harmonic carrier** (sinusoidal wave):

$$
G(x, y; \theta, f, \phi, \sigma_x, \sigma_y) = \exp\left(-\frac{x'^2}{2\sigma_x^2} - \frac{y'^2}{2\sigma_y^2}\right) \cdot \cos(2\pi f x' + \phi)
$$

Where:
- **Rotated coordinates**: $x' = x\cos\theta + y\sin\theta$, $y' = -x\sin\theta + y\cos\theta$
- $\theta$ = **orientation** (angle of edge detection, 0 to 2π)
- $f$ = **spatial frequency** (cycles per pixel, controls stripe spacing)
- $\phi$ = **phase** (shift of the wave, 0 to 2π)
- $\sigma_x, \sigma_y$ = **Gaussian spreads** (anisotropic, controls receptive field shape)

**Biological motivation**: V1 simple cells in mammalian visual cortex behave like Gabor filters - they respond to oriented edges at specific frequencies.

---

## Algorithm Implementation

### Step-by-Step Kernel Building (`_build_kernels`)

```python
def _build_kernels(self, device: torch.device) -> torch.Tensor:
    # 1. MAP UNCONSTRAINED PARAMETERS TO VALID RANGES
    theta = self.theta_u % (2 * math.pi)                              # [0, 2π]
    phase = self.phase_u % (2 * math.pi)                              # [0, 2π]
    freq = fmin + sigmoid(self.freq_u) * (fmax - fmin)                # [fmin, fmax]
    sigx = softplus(self.sigx_u) + smin                               # [smin, ∞)
    sigy = softplus(self.sigy_u) + smin                               # [smin, ∞)
    
    # 2. CREATE SPATIAL COORDINATE MESH
    # For kernel_size=31: x,y ∈ [-15, 15] (centered at 0)
    x, y = meshgrid(linspace(-r, r, k), ...)                         # [31, 31]
    
    # 3. ROTATE COORDINATES BY θ (per filter)
    cos_t = cos(theta).view(-1, 1, 1)                                 # [N, 1, 1]
    sin_t = sin(theta).view(-1, 1, 1)
    xp = cos_t * x + sin_t * y                                        # [N, 31, 31]
    yp = -sin_t * x + cos_t * y
    
    # 4. GAUSSIAN ENVELOPE (anisotropic)
    env = exp(-0.5 * ((xp/σx)² + (yp/σy)²))                          # [N, 31, 31]
    
    # 5. HARMONIC CARRIER (sinusoidal grating)
    carrier = cos(2π * f * xp + φ)                                    # [N, 31, 31]
    
    # 6. MULTIPLY: Gabor = Gaussian × Carrier
    k = env * carrier                                                 # [N, 31, 31]
    
    # 7. NORMALIZE: Zero-mean, unit L2 norm
    k = k - k.mean(dim=(1,2), keepdim=True)                          # remove DC
    k = k / (k.norm(dim=(1,2)) + ε)                                  # unit energy
    
    # 8. SCALE BY LEARNABLE AMPLITUDE
    # v1: k *= gain * gate
    # v2: k *= gate only (gain=1 fixed)
    
    return k  # [N, kernel_size, kernel_size]
```

---

## Key Differences: v1 vs v2

### Version 1 (`gabor_cnn.py`) - With Learnable Gain

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| **Amplitude** | `gain_u` (learnable) → `tanh(gain_u)` ∈ [-1, 1] | Each filter learns its own strength |
| **Gating** | `gate_u` (learnable) → `sigmoid(gate_u)` ∈ [0, 1] | Controls filter activation (quantity) |
| **Final scale** | `k *= gain * gate` | Combined amplitude and gating |
| **Sparsity loss** | `sigmoid(gate_u).abs().mean()` | L1 penalty on gates |
| **Post-processing** | Learnable affine (depthwise 1×1 conv) + SiLU | Preserves per-channel amplitude |

**Parameters per filter**: θ, f, φ, σx, σy, **gain**, gate = **7 params**

### Version 2 (`gabor_cnn_2.py`) - Fixed Gain

| Component | Implementation | Rationale |
|-----------|----------------|-----------|
| **Amplitude** | Fixed to 1.0 (removed `gain_u`) | Simplification - gate controls amplitude |
| **Gating** | `gate_u` (learnable) → `sigmoid(gate_u)` ∈ [0, 1] | Single amplitude control |
| **Final scale** | `k *= gate` | Only gating |
| **Sparsity loss** | `(g * (1-g)).mean() * 4` | **Entropy penalty** - encourages discrete 0/1 |
| **Post-processing** | SiLU only (no learnable affine) | Simpler, shared nonlinearity |

**Parameters per filter**: θ, f, φ, σx, σy, gate = **6 params** (14% reduction)

---

## Learnable Properties - Current State

### ✅ Already Learnable (Both Versions)

| Property | Parameter | Range | Mapping | What it Controls |
|----------|-----------|-------|---------|------------------|
| **Orientation** | `theta_u` | ℝ | `% (2π)` | Edge direction (0°=horizontal, 90°=vertical) |
| **Frequency** | `freq_u` | ℝ | `fmin + sigmoid(·)*(fmax-fmin)` | Stripe spacing (0.05-0.25 cycles/pixel) |
| **Phase** | `phase_u` | ℝ | `% (2π)` | Wave shift (0=cosine, π/2=sine, π=-cosine) |
| **Sigma X** | `sigx_u` | ℝ | `softplus(·) + smin` | Horizontal spread (min 3.0 pixels) |
| **Sigma Y** | `sigy_u` | ℝ | `softplus(·) + smin` | Vertical spread (min 3.0 pixels) |
| **Gate** | `gate_u` | ℝ | `sigmoid(·)` | Filter activation strength [0,1] |
| **Gain** (v1 only) | `gain_u` | ℝ | `tanh(·)` | Amplitude multiplier [-1,1] |

### 🔍 What Each Property Does Visually

**θ (Orientation)**: Rotates the stripes
```
θ=0°:     θ=45°:    θ=90°:
─────     ╲╲╲╲╲    │││││
─────     ╲╲╲╲╲    │││││
─────     ╲╲╲╲╲    │││││
```

**f (Frequency)**: Controls stripe density
```
f=0.1:    f=0.2:    f=0.3:
─────     ─────     ─────
          ─────
─────              ─────
          ─────
─────     ─────     ─────
```

**φ (Phase)**: Shifts stripes laterally
```
φ=0:      φ=π/2:    φ=π:
+─+       ─+─       ─+─
─+─       +─+       +─+
+─+       ─+─       ─+─
(+ = bright, - = dark)
```

**σx, σy (Sigmas)**: Control envelope shape
```
σx=σy:    σx>σy:    σx<σy:
  ●●●       ▬▬▬       ▌▌▌
  ●●●       ▬▬▬       ▌▌▌
  ●●●       ▬▬▬       ▌▌▌
(isotropic) (wide)   (tall)
```

---

## Advanced Learnability Questions

### Q1: Can we make frequency ranges learnable per-filter?

**Current**: All filters share `fmin=0.05, fmax=0.25` (hardcoded in `__init__`)

**Proposal**: Make each filter have its own range
```python
# Instead of:
self.fmin, self.fmax = init_freq  # shared scalar

# Use:
self.fmin_u = nn.Parameter(torch.full([out_filters], init_freq[0]))
self.fmax_u = nn.Parameter(torch.full([out_filters], init_freq[1]))

# Then in _build_kernels:
fmin = torch.sigmoid(self.fmin_u) * 0.3  # [0, 0.3]
fmax = fmin + torch.sigmoid(self.fmax_u) * (0.5 - fmin)  # [fmin, 0.5]
freq = fmin + torch.sigmoid(self.freq_u) * (fmax - fmin)
```

**Benefits**:
- Some filters could specialize in low-freq (blobs), others in high-freq (fine texture)
- More adaptive to dataset statistics
- Total params: +2 per filter = +64 for 32 filters

**Drawbacks**:
- More parameters to tune
- Harder to interpret (loses uniform frequency bands)
- May need stronger regularization

---

### Q2: Can we make sigma ranges learnable?

**Current**: `smin = 3.0` (hardcoded floor), `smax = ∞` (unbounded via softplus)

**Proposal**: Per-filter sigma bounds
```python
self.smin_u = nn.Parameter(torch.full([out_filters], 3.0))
self.smax_u = nn.Parameter(torch.full([out_filters], 8.0))

# In _build_kernels:
smin = F.softplus(self.smin_u) + 1.0  # minimum 1 pixel
smax = smin + F.softplus(self.smax_u)  # max > min
sigx = smin + torch.sigmoid(self.sigx_u) * (smax - smin)
```

**Benefits**:
- Filters could learn ultra-local (smin=1) vs global (smax=20) receptive fields
- Adaptive to image resolution

**Drawbacks**:
- Risk of collapse: smin→0 kills filter, smax→∞ makes it blob
- Need careful initialization and bounds

---

### Q3: Can we make kernel size adaptive per-filter?

**Current**: All filters share same `kernel_size=31` (spatial support)

**This is HARD** because:
1. PyTorch `F.conv2d` requires all kernels have same spatial size
2. Would need custom CUDA kernel or padding tricks
3. Dynamic shapes break batching efficiency

**Workaround**: Use sigma to control *effective* receptive field
- Large σ → filter is smooth, ignores small kernel size
- Small σ → filter is sharp, uses full 31×31

So we **already have adaptive RF via σ**! No need for variable kernel size.

---

### Q4: Can we learn separate X and Y frequencies?

**Current**: Single frequency `f` applied along rotated X-axis only (standard Gabor)

**Proposal**: 2D frequency vector
```python
self.freqx_u = nn.Parameter(torch.zeros(out_filters))
self.freqy_u = nn.Parameter(torch.zeros(out_filters))

carrier = cos(2*π * (fx*xp + fy*yp) + φ)
```

**Result**: Creates **plaid patterns** (checkerboards) instead of gratings

**Biology**: Some V2 neurons respond to plaids, but V1 simple cells are 1D gratings

**Decision**: Stick with 1D frequency (standard Gabor) for V1 modeling

---

### Q5: Can we learn asymmetric phases?

**Current**: Single phase `φ` for cosine carrier

**Proposal**: Learn sin/cos mixture (complex Gabor)
```python
self.phase_cos_u = nn.Parameter(torch.zeros(out_filters))
self.phase_sin_u = nn.Parameter(torch.zeros(out_filters))

carrier_cos = cos(2*π*f*xp)
carrier_sin = sin(2*π*f*xp)
carrier = phase_cos * carrier_cos + phase_sin * carrier_sin
```

**This is equivalent** to current phase shift, just different parameterization

**Decision**: Keep current (simpler, interpretable as "phase angle")

---

## Critical Implementation Details

### Why Normalization Matters

```python
# Step 7: Zero-mean, unit L2
k = k - k.mean(dim=(1,2), keepdim=True)
k = k / (k.norm(dim=(1,2)) + ε)
```

**Without zero-mean**: DC component → uniform brightness shift (not edge detection)
**Without unit norm**: Filters with high freq or large σ would dominate gradients

### Why We Use Unconstrained Parameters

```python
self.theta_u = nn.Parameter(torch.rand(out_filters) * 2*π)  # NOT constrained!
```

**Problem**: If we used `theta = sigmoid(theta_u) * 2π`, gradients vanish at boundaries

**Solution**: Use unconstrained ℝ → apply modulo in forward pass
- Gradients always flow (no saturation)
- Multiple rotations OK: θ=0 ≡ θ=2π ≡ θ=4π

### Spatial Attention Mechanism

```python
spatial_attn = Sequential(
    Conv2d(N, N, 3, groups=N),  # depthwise (per-filter)
    ReLU(),
    Conv2d(N, N, 1),             # pointwise (cross-filter mixing)
    Sigmoid()                    # → [0,1] attention map
)
y = gabor_maps * attention_maps  # element-wise modulation
```

**Purpose**: Let filters focus on informative regions
- Depthwise: Aggregates local context around each Gabor response
- Pointwise: Allows filter interactions (e.g., "strong horizontal AND weak vertical")
- Per-location weighting: Different weights at each (h,w) pixel

---

## Sparsity Mechanisms

### v1: L1 Penalty (Broken)

```python
sparsity_loss = sigmoid(gate_u).abs().mean()
```

**Problem**: `sigmoid(x) ∈ [0,1]`, always positive → `abs()` does nothing!
**Result**: Penalty = mean(gate), encourages gates→0 uniformly (not selective)
**Observed**: Gates stuck at 0.82 (equilibrium where CE loss = sparsity pressure)

### v2: Entropy Penalty (Fixed)

```python
g = sigmoid(gate_u)
sparsity_loss = (g * (1-g)).mean() * 4
```

**Intuition**: Binary entropy $H = -p\log p - (1-p)\log(1-p)$, linearized as $4p(1-p)$

| Gate Value | Entropy Penalty | Interpretation |
|------------|----------------|----------------|
| g = 0.0 | 0.0 | ✅ Fully OFF (no penalty) |
| g = 0.5 | 1.0 | ❌ Uncertain (maximum penalty) |
| g = 1.0 | 0.0 | ✅ Fully ON (no penalty) |

**Result**: Encourages discrete decisions (0 or 1), not soft gates (0.5)

---

## Filter Interpretability

### What Makes Gabor Filters Interpretable?

1. **Explicit parameters**: θ=45° means "detects diagonal edges" (no blackbox)
2. **Biological correspondence**: Matches V1 simple cell tuning curves
3. **Visualizable**: Can plot learned kernels directly
4. **Sparse gating**: Can identify which filters are "active" vs "pruned"

### Example Learned Filter Analysis

```python
with torch.no_grad():
    theta = model.gabor.theta_u % (2*π)
    freq = fmin + sigmoid(model.gabor.freq_u) * (fmax - fmin)
    gate = sigmoid(model.gabor.gate_u)
    
    for i in range(32):
        print(f"Filter {i}: θ={theta[i]*180/π:.1f}°, "
              f"f={freq[i]:.3f}, gate={gate[i]:.2f}, "
              f"active={'✓' if gate[i]>0.5 else '✗'}")
```

**Output example**:
```
Filter 0: θ=0.0°, f=0.150, gate=0.95, active=✓   (horizontal edges)
Filter 1: θ=45.0°, f=0.180, gate=0.89, active=✓  (diagonal edges)
Filter 2: θ=90.0°, f=0.120, gate=0.92, active=✓  (vertical edges)
Filter 3: θ=135.0°, f=0.200, gate=0.03, active=✗ (pruned!)
...
```

---

## Potential Improvements

### 1. **Learnable Frequency Ranges** (Medium Priority)
- **Change**: Per-filter `fmin`, `fmax` instead of shared
- **Benefit**: Some filters specialize in low-freq (shapes), others high-freq (texture)
- **Cost**: +2 params/filter (+64 for 32 filters)
- **Implementation**: 10 lines of code

### 2. **Learnable Sigma Bounds** (Low Priority)
- **Change**: Per-filter `smin`, `smax`
- **Benefit**: Ultra-local vs global receptive fields
- **Risk**: Instability (filters could collapse to σ→0)
- **Alternative**: Current design already allows σ adaptation via softplus

### 3. **Phase Diversity Initialization** (High Priority)
- **Current**: `phase_u` initialized to zero (all cosine)
- **Better**: Initialize to [0, π/2, π, 3π/2] for even/odd symmetry diversity
- **Benefit**: Faster convergence, better edge detection (complementary phases)
- **Implementation**: 2 lines in `__init__`

### 4. **Group-wise Frequency Bands** (Medium Priority)
- **Idea**: Divide 32 filters into 4 groups with non-overlapping frequency ranges
  - Group 1: [0.05, 0.10] - low-freq (blobs, shapes)
  - Group 2: [0.10, 0.15] - mid-low freq
  - Group 3: [0.15, 0.20] - mid-high freq
  - Group 4: [0.20, 0.25] - high-freq (fine texture)
- **Benefit**: Forced multi-scale representation
- **Implementation**: Different `fmin`/`fmax` per group in `__init__`

### 5. **Differentiable Kernel Size** (Low Priority, Hard)
- **Current limitation**: All kernels 31×31
- **Workaround**: Already handled via σ (effective RF adapts)
- **True dynamic size**: Requires custom CUDA kernel (not worth it)

---

## Conclusion

The Gabor module is **highly learnable** with 6-7 parameters per filter:
- ✅ **Orientation** (θ): Fully learnable, unbounded
- ✅ **Frequency** (f): Learnable within `[fmin, fmax]` bounds
- ✅ **Phase** (φ): Fully learnable, unbounded
- ✅ **Sigma X/Y** (σx, σy): Learnable with floor `smin`
- ✅ **Gate**: Learnable [0,1] activation strength
- ⚠️ **Gain** (v1 only): Learnable [-1,1] amplitude (removed in v2 for simplicity)

**Future work** could explore:
1. Per-filter frequency ranges (easy, useful)
2. Phase diversity initialization (easy, recommended)
3. Grouped frequency bands (medium difficulty, good for multi-scale)

The current design already provides **adaptive receptive fields** through σ learning, making additional kernel size flexibility unnecessary.
