#!/usr/bin/env python
# coding: utf-8

# # REINFORCE
# 
# Policy gradient theorem을 다시 한번 적어보자.
# 
# $$\nabla_{\theta} J(\theta) \propto \mathbb{E}_{\pi_{\theta}} \left[ Q^{\pi_{\theta}}(s, a) \cdot \nabla_{\theta} \log \pi_{\theta}(a|s) \right]. $$
# 
# <br>
# 
# 예상했겠지만, 위 그레디언트를 정확히 구하는 것은 불가능하다. 기댓값이야 Monte Carlo 기법으로 근사시킬 수 있다. 정책 $\pi_{\theta}$로 환경과 상호작용을 엄청나게 많이 해서 trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$를 많이 생성한다. 그리고 trajectory에 있는 각 $(s_t, a_t)$마다 $Q^{\pi_{\theta}}(s_t, a_t) \cdot \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$을 계산해서 표본 평균을 구하면 되기 때문이다. $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$ 계산은 딥러닝 프레임워크에서 알아서 해준다. 그런데 여전히 $Q^{\pi_{\theta}}(s_t, a_t)$를 계산하기가 어렵다. 그래서 우리는 $Q^{\pi_{\theta}}(s_t, a_t)$ 대신 다른 것으로 대체해야 한다. 이 $Q^{\pi_{\theta}}(s_t, a_t)$을 어떤 것으로 대체하느냐에 따라서 알고리즘의 이름이 달라지고 성능도 크게 달라진다. 이번 장에서는 가장 쉽고 간단한 방법인 REINFORCE 알고리즘에 대해 알아본다.
# 
# <br>
# 
# ---
# 
# ## REINFORCE: $Q^{\pi_{\theta}}(s, a)$ 대신 return 사용
# 
# 제목이 곧 내용이다. REINFORCE는 한 에피소드를 진행하여 하나의 trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$를 생성하고, 다음의 policy gradient의 추정치를 사용하여 경사하강법을 진행한다.
# 아래 식에서 실제 그레디언트 $\nabla_{\theta} J(\theta)$의 추정치를 $\hat{g}$로 표시하였다. 
# 
# $$\nabla_{\theta} J(\theta) \approx \hat{g} := \frac{1}{T} \sum\limits_{t=0}^{T} G_t \nabla_{\theta}\log\pi_{\theta}(a_t | s_t),$$
# 
# 이때, $G_t = r_t + \gamma r_{t+1} + \ldots + \gamma^{T-t} r_T$이다. REINFORCE의 알고리즘 다음과 같다.
# 
# ```{prf:algorithm} REINFORCE
# :label: REINFORCE
# 
# 1. 정책 네트워크 $\pi_{\theta}$의 파라미터 $\theta$ 초기화
# 2. for _ in range(n_episodes):
# 3. $\qquad$ $\pi_{\theta}$를 따라 에피소드를 진행하여 $\tau=(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$ 수집
# 4. $\qquad$ for $t=0, 1, 2, \ldots, T$:
# 5. $\qquad\qquad$ $G_t = r_t + \gamma r_{t+1} + \ldots + \gamma^{T-t} r_{T}$
# 6. $\qquad\qquad$ $\hat{g}_t = G_t \nabla_{\theta} \log \pi_{\theta} (a_t|s_t)$
# 7. $\qquad$ $\hat{g} = \frac{1}{T}\sum\limits_{t=0}^{T}\hat{g}_{t}$
# 8. $\qquad$ $\theta \leftarrow \theta + \eta \hat{g}$  $\quad$ # Gradient ascent
# ```
# 
# <br>
# 
# 필자는 심층강화학습을 책으로 공부하면서 {prf:ref}`REINFORCE`의 $\nabla_{\theta} \log \pi_{\theta} (a_t|s_t)$이 어떻게 코드로 구현되는지 정말 궁금했다.
