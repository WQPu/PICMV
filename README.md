## MM for Covariance Estimation 
Optimization Problem: 

$$
\begin{align}
\min_{\{\sigma_i \}}\ &\log\det(\mathbf{R})+\textrm{Tr}(\mathbf{S}\mathbf{R}^{-1})\quad \\
\textrm{s.t.}\ \ \ &\mathbf{R}=\sum_{i=1}^K \sigma_i \mathbf{a}_i\mathbf{a}_i^H+\sigma_n \mathbf{I},\\
&\sigma_i\geq \epsilon,\forall i.
\end{align}
$$
- $\mathbf{S}=\frac{1}{N}\sum_{i=1}^N \mathbf{s}_i \mathbf{s}_i^H$ is a given PSD matrix, $N$ is number of samples
- $\mathbf{a}_i$ is known, $\sigma_n$ is a known constant

Prior knowledge: most $\sigma_i$ are close to zero (not exactly equal to zero) while only few are large. To enforce this information, introduce a sufficiently small constant $\epsilon>0$ to ensure $\sigma_i\geq \epsilon$,  add l1-regularization term $\sum_i |\sigma_i - \epsilon|$ to objective. Since $\sigma_i\geq \epsilon$, $\sum_i |\sigma_i - \epsilon|$ becomes $\sum_i (\sigma_i - \epsilon)$. 

For notation simplicity, denote $\mathbf{P}=\text{diag}\{\sigma_1,\ldots,\sigma_K \}$, $\mathbf{A}=[\mathbf{a}_1,\ldots,\mathbf{a}_K]$, thus $$\mathbf{R}=\mathbf{A}\mathbf{P}\mathbf{A}^H+\sigma_n\mathbf{I}$$
By Woodbury matrix identity, we have 
$$
\mathbf{R}^{-1}=\sigma_n^{-1}\mathbf{I}-\sigma_n^{-2}\mathbf{A}(\mathbf{P}^{-1}+\sigma_n^{-1}\mathbf{A}^H\mathbf{A})^{-1}\mathbf{A}^H
$$

In the rest part we use notations: $\mathbf{X}=(\mathbf{P}^{-1}+\sigma_n^{-1}\mathbf{A}^H\mathbf{A}),\ \mathbf{X}_t=(\mathbf{P}_t^{-1}+\sigma_n^{-1}\mathbf{A}\mathbf{A}^H)$


#### Upper bound function of $\textrm{Tr}(\mathbf{S}\mathbf{R}^{-1})$
By Lemma 1, we have 
$$
\begin{align}
\textrm{Tr}(\mathbf{S}\mathbf{R}^{-1})
&=\sigma_n^{-1}\textrm{Tr}(\mathbf{S})-\sigma_n^{-2}\textrm{Tr}(\mathbf{A}(\mathbf{P}^{-1}+\sigma_n^{-1}\mathbf{A}\mathbf{A}^H)^{-1}\mathbf{A}^H\mathbf{S})\\
&\leq \sigma_n^{-1}\textrm{Tr}(\mathbf{S})+\sigma_n^{-2}\left[\operatorname{Tr}\left(\mathbf{X}_t^{-1} \mathbf{Y} \mathbf{X}_t^{-1}\left(\mathbf{X}-\mathbf{X}_t\right)\right)-\operatorname{Tr}\left(\mathbf{Y} \mathbf{X}_t^{-1}\right)\right]
\end{align}
$$
where 
$$
\mathbf{Y}=\mathbf{A}^H\mathbf{S}\mathbf{A},\ \mathbf{X}=(\mathbf{P}^{-1}+\sigma_n^{-1}\mathbf{A}^H\mathbf{A}),\ \mathbf{X}_t=(\mathbf{P}_t^{-1}+\sigma_n^{-1}\mathbf{A}\mathbf{A}^H)
$$


Ignoring constant, we have 
$$
\begin{align}
\textrm{Tr}(\mathbf{S}\mathbf{R}^{-1})
&\leq \sigma_n^{-2}\operatorname{Tr}(\mathbf{S}\mathbf{A}\mathbf{X}_t^{-1}\mathbf{P}^{-1}\mathbf{X}_t^{-1}\mathbf{A}^H)+\text{const}
\end{align}
$$

#### Upper bound function of $\log\det(\mathbf{R})$
By Lemma 2, we have 
$$
\begin{align}
\log \operatorname{det}(\mathbf{R}) &=\log \operatorname{det}(\mathbf{A}\mathbf{P}\mathbf{A}^H+\sigma_n\mathbf{I})\\
& = \log \operatorname{det}((\mathbf{A}^H\mathbf{A}\mathbf{P}+\sigma_n\mathbf{I}))\ (\text{Weinstein–Aronszajn identity})\\
&=\log \operatorname{det}((\sigma_n^{-1}\mathbf{A}^H\mathbf{A}+\mathbf{P}^{-1})\mathbf{P}\sigma_n)\\
&=\log \operatorname{det}(\sigma_n^{-1}\mathbf{A}^H\mathbf{A}+\mathbf{P}^{-1})+\log \operatorname{det}(\mathbf{P})+\text{const}\\

&\leq \operatorname{Tr}\left(\mathbf{X}_t^{-1}\mathbf{P}^{-1}\right)+\log \operatorname{det}(\mathbf{P})+\text{const}
\end{align}
$$

where the last inequality is due Lemma 2 with $\mathbf{X}=\sigma_n^{-1}\mathbf{A}^H\mathbf{A}+\mathbf{P}^{-1}$

#### MM Solution
For $\sigma_n^{-2}\operatorname{Tr}(\mathbf{S}\mathbf{A}\mathbf{X}_t^{-1}\mathbf{P}^{-1}\mathbf{X}_t^{-1}\mathbf{A}^H)$, define $\mathbf{b}_i=\sigma_n^{-1}\mathbf{X}_t^{-1}\mathbf{A}^H\mathbf{s}_i$, then 
$$
\sigma_n^{-2}\operatorname{Tr}(\mathbf{S}\mathbf{A}\mathbf{X}_t^{-1}\mathbf{P}^{-1}\mathbf{X}_t^{-1}\mathbf{A}^H)=\frac{1}{N}\sum_{i=1}^N \mathbf{b}_i^H \mathbf{P}^{-1}\mathbf{b}_i=\sum_{k=1}^K \sigma_k^{-1}\left(\frac{1}{N}\sum_{i=1}^N|\mathbf{b}_{ik}|^2\right)
$$
where $\mathbf{b}_{ik}$ is the $k$th component of $\mathbf{b}_{i}$

For $\operatorname{Tr}\left(\mathbf{X}_t^{-1}\mathbf{P}^{-1}\right)+\log \operatorname{det}(\mathbf{P})$, we have 
$$
\operatorname{Tr}\left(\mathbf{X}_t^{-1}\mathbf{P}^{-1}\right)+\log \operatorname{det}(\mathbf{P})=\sum_{k=1}^K [\mathbf{X}_t^{-1}]_{kk}\sigma_k^{-1}+\log \sigma_k
$$
Hence, MM surrogate function is separable over $k$,  for each $\sigma_k$, the subproblem is 

$$
\min_{\sigma_k\geq 0}\ \left([\mathbf{X}_t^{-1}]_{kk}+\left(\frac{1}{N}\sum_{i=1}^N|\mathbf{b}_{ik}|^2\right)\right) \sigma_k^{-1}+\log \sigma_k
$$
Thus $\sigma_k^{t+1}=\left([\mathbf{X}_t^{-1}]_{kk}+\left(\frac{1}{N}\sum_{i=1}^N|\mathbf{b}_{ik}|^2\right)\right)$


Lemma 1: Function $\operatorname{Tr}\left(\mathbf{Y X}^{-1}\right)$ with both $\mathbf{Y}$ and $\mathbf{X}$ in $\mathbb{S}_{++}$ can be lower bounded as
$$
\operatorname{Tr}\left(\mathbf{YX}^{-1}\right) \geq \operatorname{Tr}\left(\mathbf{Y} \mathbf{X}_t^{-1}\right)-\operatorname{Tr}\left(\mathbf{X}_t^{-1} \mathbf{Y} \mathbf{X}_t^{-1}\left(\mathbf{X}-\mathbf{X}_t\right)\right)
$$


Lemma 2: At any $\mathbf{X}_t \succ \mathbf{0}$ of dimension $M$, since $\log \operatorname{det}(\cdot)$ is concave, $\log \operatorname{det}(\mathbf{X})$ can be upper bounded by its first order Taylor expansion at $\mathbf{X}_t$ :
$$
\log \operatorname{det}(\mathbf{X}) \leq \log \operatorname{det}\left(\mathbf{X}_t\right)+\operatorname{Tr}\left(\mathbf{X}_t^{-1} \mathbf{X}\right)-M
$$
with equality achieved at $\mathbf{X}_t$.



