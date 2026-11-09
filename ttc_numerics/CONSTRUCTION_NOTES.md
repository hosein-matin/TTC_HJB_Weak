# Complete-grid construction: geometry, proof, and computational scope

This note documents `construct.py` and `construction_experiment.py`. All
coordinates below have the order `z=(s,Xi,Lambda,zeta_L)`. The numerical
experiment uses the supplied physical prediction model with `alpha=2`,
`TH=1`, and leader acceleration `(.5,-1.2,.8,-.4,0)` on the five successive
intervals of length `.2`. The optimal velocity is

\[
\mathcal V(x)=\tanh(x-2)+\tanh 2.
\]

The fixed approximation and initial boxes are

\[
\begin{split}
B_c&=[.15,.27]\times[.05,.12]\times[-1.4,-1.1]\times[.27,.52],\\
D_c&=[.15,.25]\times[.08,.12]\times[-1.4,-1.2]\times[.30,.50].
\end{split}
\]

The stopping duration is `h=.02`. Grid geometry is fixed in advance;
every node is populated. Held-out query locations never select the grid
supports. The fixed slice is `s=.20`, `zeta_L=.40`, with the full `Xi` and
`Lambda` ranges of `B_c`.

## 1. Uniform contact and transversality on the complete box

Write elapsed time as `k`, closing speed as `w=-Lambda`, and
`A(t)=integral_0^t a_L(u) du`. The prescribed primitive has range
`[-.14,.10]`. Therefore, for every trajectory starting in `B_c`,

\[
.03\le\zeta_L(k)\le .76
\]

as long as the absolute time is in `[0,1]`. Before contact,
`Xi>=0` implies `Vopt(Xi)>=0`. Since `a_L<=.8`,

\[
w'=-a_L+2\mathcal V(\Xi)-2\zeta_L-2w
\ge-2.32-2w.
\]

Initial closing speed is at least `1.1`. The integrating-factor comparison
therefore gives

\[
w(k)\ge 2.26e^{-2k}-1.16.
\]

Set

\[
c=.3,\qquad T_*={1\over2}\log{2.26\over1.46}
             =0.2184641887819747.
\]

The lower bound is at least `c` for `0<=k<=T_*`. If no contact had occurred
by `T_*`, the gap would have decreased by at least

\[
\int_0^{T_*}(2.26e^{-2k}-1.16)\,dk
=.4-1.16T_*=0.14658154101290938>.12,
\]

contradicting its positive initial gap. Hence every point of `B_c`
contacts before `T_*`, with `Lambda<-c`. The argument first establishes
positive closing speed, so the precontact gap stays in `[0,.12]`.

On that gap interval, `a_L>=-1.2` and `zeta_L>=.03` also give

\[
w'\le1.14+2\mathcal V(.12)-2w.
\]

Its equilibrium is below `1.1`, so `w<=1.4` up to contact. Consequently

\[
.05/1.4\le V(z)<T_*,\qquad z\in B_c.
\]

All these contacts precede the horizon because `.27+T_*<1`.
The initial follower speed `v_F=zeta_L+w` lies in `[1.37,1.92]`.
Its equation `v_F'=2(Vopt(Xi)-v_F)` preserves nonnegativity and an upper
bound of `1.92` up to contact. This is below the supplied sampling cap
`2 tanh 2`. That sampling cap is not the global supremum of `Vopt`.

### Inclusion of the short forward tube

Every point in `D_c` has `V>=.08/1.4>.02`. Before time `.02`,

- `s+k` lies in `[.15,.27]`;
- the gap lies in `[.08-1.4*.02,.12]=[.052,.12]`;
- the leader speed lies in `[.30-1.2*.02,.50+.8*.02]=[.276,.516]`;
- `w<=1.4` as above.

For the lower bound on `w`, the coarser leader-speed estimate from
`D_c` is `zeta_L<=.74`. Thus

\[
w(k)\ge(1.2+1.14)e^{-2k}-1.14>1.1,
\qquad0\le k\le.02.
\]

The complete short tube is therefore contained in `B_c`. This is a direct
tube inclusion, not a claim that `B_c` is an Euclidean thickening of
`D_c` of some specified radius. Only this local tube and the local stopped
residual are claimed for the construction experiment.

## 2. A finite-difference Lipschitz bound, including starting time

This argument does not differentiate the discontinuous acceleration.
Take any two initial points of `B_c`. Compare their trajectories at equal
elapsed time `k` up to the earlier of their two contacts. Both gap
trajectories then lie in `[0,.12]`. Write differences with `Delta`, and set

\[
R_s(u)=A(s+u)-A(s),\qquad
b_k={1-e^{-2k}\over2},\qquad
\ell=\operatorname{sech}^2(1.88)=0.08894480607878208.
\]

Here `ell` bounds `|Vopt'|` on `[0,.12]`. Since `A` is Lipschitz with
constant `a_infty=1.2`,

\[
|R_{s_1}(u)-R_{s_2}(u)|\le2a_\infty|s_1-s_2|.
\]

The follower filter has the exact variation-of-constants representation

\[
v_F(k)=e^{-2k}v_F(0)+
2\int_0^k e^{-2(k-v)}\mathcal V(\Xi(v))\,dv.
\]

Integrating `Xi'=zeta_L-v_F`, using
`Delta v_F(0)=Delta zeta_L(0)-Delta Lambda(0)`, and exchanging the order
of the finite integrals gives

\[
\begin{split}
\Delta\Xi(k)={}&\Delta\Xi_0+b_k\Delta\Lambda_0
 +(k-b_k)\Delta\zeta_0
 +\int_0^k[R_{s_1}(u)-R_{s_2}(u)]\,du\\
&-\int_0^k(1-e^{-2(k-v)})
 [\mathcal V(\Xi_1(v))-\mathcal V(\Xi_2(v))]\,dv.
\end{split}
\]

Both `b_k` and `k-b_k` are nonnegative increasing functions of `k`.
Also `1-exp(-2(k-v))<=2(k-v)`, whose integral is `k^2`. Let
`E=sup |Delta Xi(k)|` up to the earlier contact and let `b=b_{T_*}`.
Then

\[
(1-\ell T_*^2)E\le |\Delta\Xi_0|+b|\Delta\Lambda_0|
 +(T_*-b)|\Delta\zeta_0|+2a_\infty T_*|\Delta s|.
\]

The denominator is positive:
`1-ell*T_*^2=0.9957549666598638`. At the earlier contact, the other
trajectory's nonnegative remaining gap is at least `c` times its
remaining contact time, because its closing speed stays at least `c`.
Thus the contact-time difference is at most `E/c`. Cauchy--Schwarz yields
the global Euclidean-coordinate Lipschitz bound on `B_c`

\[
|V(z_1)-V(z_2)|\le L|z_1-z_2|_2,\qquad
L={\sqrt{(2a_\infty T_*)^2+1+b^2+(T_*-b)^2}
       \over c(1-\ell T_*^2)}
 =3.828441530908998.
\]

This is an analytical bound; no observed finite-difference slope is used
as a substitute for a uniform Lipschitz constant.

## 3. Complete-grid interpolation bound

For maximum mesh parameter `h0`, coordinate `i` is divided into
`n_i=ceil((B_hi[i]-B_lo[i])/h0)` equal segments of actual length `h_i`.
All `J=product_i(n_i+1)` grid nodes are used. The univariate nodal hats
are `sigma(1-|x_i-node_i|/h_i)`; their tensor products form a partition
of unity on `B_c`, including its boundary.

If the exact coefficients are `V_j=V(z_j)`, the multilinear interpolant
`I_*` satisfies

\[
\|I_*-V\|_{L^\infty(B_c)}
\le {L\over2}\sqrt{\sum_{i=1}^4h_i^2}.
\]

Indeed, at a query point the interpolation weights are probabilities on
the surrounding vertices. Jensen's inequality bounds the weighted
distance by the square root of its second moment. The coordinate second
moments are `h_i^2 u_i(1-u_i)<=h_i^2/4`.

The maximum meshes `.02`, `.01`, and `.005` give respectively `7,840`,
`83,824`, and `1,166,625` complete nodes. The `.01` and `.005` time meshes
contain the forcing jump at `s=.2`; the `.02` time mesh does not. The
three empirical errors therefore should not be interpreted as a
universal convergence-rate fit.

## 4. Explicit support-preserving ReLU multiplication

Let `sigma(t)=max(t,0)` and

\[
g(t)=2\sigma(t)-4\sigma(t-.5)+2\sigma(t-1),\qquad
g_j=\underbrace{g\circ\cdots\circ g}_{j\text{ factors}}.
\]

The square network

\[
q_k(t)=t-\sum_{j=1}^k4^{-j}g_j(t)
\]

is the piecewise affine interpolant of `t^2` on the dyadic grid of
spacing `2^{-k}` in `[0,1]`. This follows inductively: at each new
midpoint, `4^{-j}g_j` removes exactly the previous interpolant's
midpoint error and vanishes at the old grid points. On each final
interval `[a,b]`, the interpolation error is `(t-a)(b-t)`. Hence

\[
0\le q_k(t)-t^2\le e_k:=4^{-(k+1)}.
\]

Define the actual implemented product block

\[
M_k(x,y)=\sigma(p)-\sigma(p-1),\quad
p=q_k((x+y)/2)-q_k(|x-y|/2),
\quad |u|=\sigma(u)+\sigma(-u).
\]

For `x,y` in `[0,1]`, both square arguments lie in `[0,1]`. The
difference of their two square errors lies in `[-e_k,e_k]`; projection
onto `[0,1]` cannot increase distance from `xy`. Therefore

\[
0\le M_k(x,y)\le1,\qquad |M_k(x,y)-xy|\le e_k,
\qquad M_k(x,0)=M_k(0,y)=0.
\]

The zero identities are exact algebraic identities, not merely small
approximation errors. A sequential product `P_{k,d}` of `d` factors
using `d-1` such blocks has range `[0,1]`, preserves every zero factor,
and has error at most `(d-1)e_k`, by induction.

For each grid node, replace its exact four-factor tensor hat by
`P_{k,4}` applied to its four univariate hats. The resulting finite sum
with nodal coefficients is an explicitly specified global ReLU map.
Because zero factors are preserved, only the at most `2^4=16` active
tensor hats contribute at any point of `B_c`. With exact nodal values,

\[
\|N_*-I_*\|_{L^\infty(B_c)}\le16T_*\,3e_k.
\]

The code uses `k=ceil(log2(1/h0))`, namely `6,7,8` for the three grids.
It evaluates the actual approximate products. Its support-aware path
uses cell indices solely to omit known zero contributions. The literal
`full_sum` path evaluates all global hats and product blocks, including
for points outside the box. Numerical checks compare the paths on the
coarse complete grid and verify exact zeros outside the union of all
supports. This acceleration is not a claim about materialized dense
network inference latency.

### Gate and layer counts

The directly evaluated computational graph permits affine skip edges.
It has three ReLU gates per univariate hat, and `6k+4` per product
block: two for the absolute value, `6k` for the two square recurrences,
and two for clipping. Without sharing blocks across nodes, the four
dimensional network therefore has

\[
J[12+3(6k+4)]=J(18k+24)
\]

ReLU gates, and longest ReLU depth `2+3(k+2)=3k+8`. Affine output
nodes, weights, and biases are not counted as ReLU gates. This is not
a trainable-parameter count.

For completeness, an ordinary strictly layered realization can insert
nonnegative identity carries. A multiplier can be scheduled as follows:

1. One layer of three gates forms `sigma((x+y)/2)`, `sigma(x-y)`, and
   `sigma(y-x)`. The two square inputs are affine functions of these.
2. The first square-tent layer needs six gates, three per square branch.
   At this layer `q_0(t)=t`, so the gate `sigma(t)` also carries the
   accumulator; an extra accumulator gate is unnecessary.
3. Each of the remaining `k-1` square-tent layers needs eight gates:
   three tent gates and one `sigma(q_{j-1})` accumulator carry per
   branch. The carry is exact because each square interpolant is
   nonnegative on `[0,1]`.
4. The final clipping layer uses two gates.

This gives `3+6+8(k-1)+2=8k+3` gates over `k+2` layers. In a four-factor
chain, the unused factors require `(2+1+0)(k+2)` further identity
carries. Including the hats, the count is `J(27k+27)`, with the same
depth `3k+8`. The primary reported count is the directly evaluated DAG
gate count, with its affine-skip convention stated explicitly.

## 5. Nodal errors and the meaning of the analytical columns

The computed grid coefficients are floating-point RK4 first-contact
estimates. If their *uniform nodal error* is at most `epsilon_node`,
the use of these coefficients instead of exact nodal TTC values changes
the ReLU map by at most

\[
\epsilon_{\rm node}(1+48e_k).
\]

To see this, all approximate hat products are nonnegative, and their
sum is at most `1+16*3e_k`. This proof holds for any coefficient
perturbations with the stated bound, not only random perturbations.
Combining the previous estimates gives, in exact real arithmetic,

\[
\|\widehat N-V\|_{L^\infty(B_c)}
\le \underbrace{{L\over2}\|h\|_2+48T_*e_k}_{\delta_*}
   +\epsilon_{\rm node}(1+48e_k).
\]

Because `D_c` and its `.02` landing points lie in `B_c`, the exact
semigroup identity gives the corresponding local stopped residual
bound `2[delta_*+epsilon_node(1+48e_k)]`.

**The numerical experiment does not obtain a validated uniform bound
on `epsilon_node` or floating-point evaluation roundoff.** The columns
`exact_node_interpolation_bound`, `exact_node_relu_bound`, and
`exact_node_local_residual_bound` refer only to exact nodal values and
real arithmetic. Independent DOP853 comparisons and RK4 step refinement
are finite empirical checks. Their observed maxima are reported
separately, never substituted for a proved uniform nodal error.

The deliberate nodal perturbation experiment uses `eta=.01` and checks
the proven stability factor on independent held-out inputs. It does
not turn the original solver error into a certified bound.

## 6. Reproduction and comparison protocol

Run from this directory:

```sh
python3 construction_experiment.py --meshes 0.02 0.01 0.005 --n-test 10000 --out out
```

The script stores complete grid arrays, independent query/reference
arrays, the common 121 by 121 slice, reference checks, coefficient and
query hashes, and all results in `out/construction_results.json`.
The five fixed random seeds are recorded in that JSON. `--reuse-grids`
reuses saved grids only after verifying their geometry.

The unchanged supplied MSE and residual-regularized checkpoints are
evaluated on the same `B_c`, `D_c`, tube, landing, and slice points as
the construction. These are local common-domain comparisons. The
models were trained on the larger supplied physical sampling domain;
the construction obtains all its nodal values directly from the ODE
solver on its chosen local box. The experiment does not assert equal
training budgets or a global error guarantee for the learned models.

The report separates exact multilinear interpolation, actual ReLU
realization, the difference between them, and unchanged learned model
errors. All empirical maxima are maxima over the named finite samples.
