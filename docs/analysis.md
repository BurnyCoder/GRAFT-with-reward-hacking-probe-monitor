# Estimands, optimization and uncertainty

The [protocol](protocol.md) fixes decisions; [analysis.py](../experiments/qwen_probe/analysis.py) implements the statistics; [results.json](../reports/qwen_probe/results.json) contains the observed estimates. This document explains their meaning and mathematical limits rather than changing the analysis plan.

## What the training comparison changes

For trajectory $i$, let $A_i$ be its assistant target-token positions. The implemented SFT objective averages token cross-entropy within each trajectory, then averages the 48 trajectory losses in an accumulation window:

$$L_i=-\frac{1}{|A_i|}\sum_{t\in A_i}\log p_\theta(x_{it}\mid x_{i,<t}),\qquad L_B=\frac{1}{48}\sum_{i\in B}L_i.$$

This weights trajectories equally; it is not one token-weighted mean across the concatenated batch. The corrected pilot delegates the single accumulation division to `Accelerator.backward`. Prompt tokens have label `-100`; the collator regression verifies that only assistant targets contribute. Sources: [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation), [Qwen3 loss API](https://huggingface.co/docs/transformers/main/model_doc/qwen3), and the [actual trainer](../sft/scripts/gr/train_gr.py).

The frozen base MLP is augmented with two gated MLP branches, each of width 64. The scale, initialization, routing and optimizer come from the [released launcher](../sft/scripts/launch_gr_32b_split.sh) and [adapter implementation](../sft/scripts/gr/adapter.py). A flagged trajectory has both branches active in its forward pass, zero retain gradient and a forget gradient multiplied by two. Unflagged non-anchor trajectories update both branches. Anchors disable the forget branch in the forward pass. Their identities can change with the monitor's unflagged pool. No intervention disables this routing and anchor treatment.

For an active adapter, write the accumulated routed gradient as $g_m$ and the captured natural gradient as $g_v$. The source [split-moment optimizer](../sft/scripts/gr/split_moment.py) updates FP32 moments as

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_m,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)(c_Ag_v)^2.$$

Squaring is elementwise. The participation factor $c_A$ and role-activity gates are set by the source trainer. Bias-corrected moments produce the AdamW update, with the ordinary decoupled weight decay. Both gradient sources receive the same clipping factor. This separates the multiplier in the first moment from its usual cancellation through the second moment. The logged realized-step diagnostic averages coordinates, including zero-gradient coordinates; it need not equal two on every mixed-data update. The source regression establishes the intended multiplier in controlled cases. Filtering uses ordinary AdamW and zeroes flagged loss, so its comparison changes more than routing alone.

## Detector validation is a separate estimand

Each frozen monitor scores the same normalized trajectory content. Its strict cutoff is selected on training-side clean calibration trajectories. The heldout set selects at most one baseline-clean and one rewarded-hack record per task. Thus precision describes this selected class mixture; it does not estimate precision at an unknown deployment prevalence. The behavioral audit counts attempts even when they fail, whereas the source positive validation class contains rewarded hacks. These are distinct endpoints.

AUROC is the probability that a random labeled positive ranks above a random labeled negative, counting tied scores by halves. Thresholded precision, recall and FPR use their usual confusion-matrix denominators. Missing scores are excluded and counted; undefined precision is `null`, not zero. The implementation uses [scikit-learn AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). Calibration FPR is a selected empirical quantity and is not a population guarantee.

Whole tasks, including both selected labels where present, are resampled for detector intervals. The paired comparison resamples the same tasks for both monitors and subtracts their statistics. It conditions on the already frozen calibration cutoffs; it does not include uncertainty from drawing a new calibration set or training a new upstream detector. Bootstrap draws with undefined precision are omitted for that metric, and the paired report records the number of valid draws.

## Behavioral estimates and ambiguity

For a known binary outcome, the reported interval is the [Wilson score interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). If $s$ successes and $u$ unknown judgments occur among $N$ available episodes, the full-denominator ambiguity bounds are $[s/N,(s+u)/N]$. Entirely missing episodes remain coverage gaps; they are never silently assigned benign labels.

For each shared task, define $D_i=Y_i^{\mathrm{treatment}}-Y_i^{\mathrm{control}}$. The point estimate is the mean of known paired differences. The paired percentile bootstrap resamples tasks and reports its 2.5th and 97.5th percentiles. Ambiguous paired labels also yield worst-case lower and upper differences. The exact descriptive discordance test conditions on tasks with $D_i\ne0$ and tests their signs against probability one half. No multiple-comparison adjustment is claimed for secondary contrasts.

Eight selected tasks are not an independent random sample of all deployment tasks. Task difficulty, deterministic local feasibility selection and the shared training seed limit extrapolation. A zero-width bootstrap interval from identical outcomes is not evidence of zero population uncertainty. In particular, a point difference above the minus-five-percentage-point capability tolerance is not a powered noninferiority conclusion. Zero baseline hacking or uniformly failed tasks leaves mitigation effectiveness inconclusive by design.

## Prospective sample and seed sensitivity

Let the desired absolute reduction be $\delta=p_0-p_1=0.10$, with two-sided $\alpha=0.05$ and target power $0.80$. Let $z_a=\Phi^{-1}(0.975)$, $z_b=\Phi^{-1}(0.80)$ and $\bar p=(p_0+p_1)/2$. The independent-proportion normal approximation implemented here is

$$n_{\mathrm{per\ arm}}=\left\lceil\frac{\left[z_a\sqrt{2\bar p(1-\bar p)}+z_b\sqrt{p_0(1-p_0)+p_1(1-p_1)}\right]^2}{\delta^2}\right\rceil.$$

This uses pooled null variance and unpooled alternative variance, as described by [statsmodels' two-proportion power calculation](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.power_proportions_2indep.html). The boundary case $p_1=0$ is especially approximate. These are prospective planning numbers, not post-hoc achieved power.

For paired tasks, let $q=P(D_i\ne0)$. Since $E[D_i^2]=q$ and $E[D_i]=-\delta$, $\operatorname{Var}(D_i)=q-\delta^2$. The planning approximation is

$$n_{\mathrm{pairs}}=\left\lceil (z_a+z_b)^2\frac{q-\delta^2}{\delta^2}\right\rceil.$$

For sensitivity to $K$ independent training seeds with an assumed between-seed SD $\sigma$, the simplified model is $\operatorname{Var}(\bar D)=[\sigma^2+(q-\delta^2)/n]/K$. Solving against target variance $V_*=\delta^2/(z_a+z_b)^2$ gives

$$n\ge\frac{q-\delta^2}{K V_*-\sigma^2}.$$

A nonpositive denominator is reported as `null`: more tasks alone cannot meet this approximation at that seed count and assumed seed variation. This pilot cannot estimate $\sigma$ from one seed. Reusing the same tasks across seeds can introduce correlation, and a small number of seeds also makes normal critical values optimistic. A follow-up should measure these variance components and plan a crossed task/seed design before claiming adequate power. The known sensitivity of empirical deep-learning comparisons to seeds is discussed by [Henderson et al., Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560); that paper is motivation for caution, not a variance estimate for this SFT study.
