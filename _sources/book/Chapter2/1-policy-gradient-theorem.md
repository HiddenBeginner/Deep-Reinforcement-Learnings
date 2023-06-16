# Policy Gradient Theorem

강화학습에서 정책 (policy)은 주어진 상태에서 어떤 행동을 취할지를 알려주는 일종의 지침서 같은 것이다. 보다 더 일반적으로는, 정책 $\pi$는 주어진 상태 $s \in \mathcal{S}$에서 어떤 행동 $a \in \mathcal{A}$을 선택할 조건부 확률이다. 즉, $\pi(a | s) = \text{Pr} \left[ A_t = a | S_t = s \right]$ 이다. 만약 상태의 개수와 행동의 개수가 적다면 사람이 직접 각 $(s, a)$마다 확률을 부여하여 정책을 만들 수 있을 것이다. 하지만, 대부분의 환경은 가능한 상태와 행동의 개수가 굉장히 많으며, 심지어 부여할 수 있는 확률 값도 정말 무수히 많을 것이다. 이런 고생을 덜고자 매개변수화된 함수로 정책을 모델링하여 좋은 정책을 찾는 방법을 **policy-based** 방법이라고 한다. 매개변수를 $\theta \in \mathbb{R}^{d}$이라고 하면, 이제 매개변수화된 정책은 다음과 같이 적어줄 수 있다.

$$\pi_\theta(a | s) = \text{Pr} \left[ A_t = a | S_t =s,\theta_t=\theta \right].$$

<br>

매개변수의 값에 따라 정책의 성능이 좋을 수도 있고 나쁠 수도 있을 것이다. 우리의 목표는 좋은 정책을 만드는 매개변수를 찾는 것이다. 그러기 위해선 정책의 성능을 평가하는 성능 지표 (performance measure)가 필요하다. 매개변수에 따라 정책의 성능이 달라지므로 성능 지표는 매개변수 값에 의해 결정된다. 따라서 성능 지표를 매개변수에 대한 함수 $J(\theta)$로 적어준다. 

<br>

우리는 성능 지표를 크게 만들어주는 매개변수를 찾기 위해 매개변수에 대한 성능 지표의 그레디언트를 사용할 것이다. 

$$\theta_{\text{new}}=\theta_{\text{old}}+\alpha\widehat{\nabla}_{\theta}{J(\theta_{\text{old}})}$$

<br>

실제 그레디언트 $\nabla_{\theta} J(\theta\_{\text{old}})$을 찾을 수 있으면 베스트이지만, 일반적으로는 그레디언트에 대한 stochastic 추정치 $\widehat{\nabla}_{\theta}{J(\theta_{\text{old}})}$를 사용한다. $\widehat{\nabla}_{\theta}{J(\theta_{\text{old}})}$의 기댓값이 실제 그레디언트 $\nabla_{\theta}{J(\theta_{\text{old}})}$에 근사하는 추정량을 사용해야 할 것이다. 이와 같이 그레디언트를 사용하여 좋은 정책을 학습하는 방법을 **policy gradient** 방법이라고 부른다. 

<br>

---

## Policy Gradient Theorem
우리는 정책의 성능을 평가하는 지표 $J(\theta)$의 그레디언트를 사용하여 점점 더 좋은 정책을 찾아나갈 것이다. 그럼, 가장 먼저 성능 지표 $J(\theta)$를 정의해야 한다. 이 성능 지표는 주어진 MDP의 설정에 따라 달라질 수 있다. 성능 지표가 달라지면, 그레디언트도 달라질 것이다. 그럼 우리는 성능 지표를 정의할 때마다 그레디언트를 해석적으로 (analytically, 직접 식을 전개하여 푸는 것을 의미) 계산을 해야 하는가? 정말 다행히도 policy gradient theorem은 다양한 성능 지표에 대해서 그레디언트들이 서로 비례한다는 것을 보였다.

<br>

Policy gradient theorem을 조금 더 쉽게 기술하기 위해 주어진 MDP가 유한 상태 공간, 유한 행동 공간 갖는다고 가정할 것이다. 생각해볼 수 있는 가장 자연스러운 정책 평가 지표는 에피소드 동안 받은 보상의 총합의 기댓값일 것이다. 즉, 초기 상태의 가치 함수이다. 초기 상태 확률 분포에 따라 초기 상태가 다양하게 있을 수 있으므로 기댓값을 취하는 것이 좋을 것이다.

$$J(\theta):= \mathbb{E}_{S_0 \sim d_0} \left[ V^{\pi_{\theta}}(S_0) \right].$$ (objective)

<br>

자, 이제 식 {eq}`objective`의 그레디언트를 계산해보자. 사실, 썩 쉬워보이지 않는다. 우선, $J(\theta)$는 정책이 취하는 행동에 따라 달라질 수 있다. 그리고, 정책을 따랐을 때 방문하는 상태들에 따라서도 달라질 수 있다. 그래, 정책은 $\theta$에 대한 함수니깐 그레디언트를 구할 수 있을 것이다. 하지만 정책이 방문한 상태들의 분포는 정책 뿐만 아니라 환경의 transition 모델에 따라 달라질 수 있기 때문에 그레디언트를 계산하는 것이 만만치 않을 것이다. 파라미터 $\theta$와 성능 지표 $J(\theta)$의 관계를 도식화해보면 다음과 같을 것이다. 

<br>

정말 다행히도 식 {eq}`1step_state_value_function`의 그레디언트를 다음과 같이 쉽게 구할 수 있다는 이론이 **policy gradient theorem**이다. 

$$\nabla_{\theta} J(\theta) \propto \sum_s d_{\pi_{\theta}}(s) \sum_{a} Q^{\pi_{\theta}}(s,a) \nabla_{\theta} \pi_{\theta}(a|s),$$ (policy-gradient-theorem)

<br>

여기서 $d_{\pi_{\theta}}(s)$는 정책 $\pi_{\theta}$를 따랐을 때 상태 $s$에 머무를 확률로 이해하면 된다 (증명에 더 상세히 정의된다). 식 {eq}`policy-gradient-theorem`는 여전히 복잡해 보이지만, 우려와 다르게 방문한 상태들의 분포 $d_{\pi_{\theta}}(s)$ 를 미분하는 일은 발생하지 않았다. $d_{\pi_{\theta}}(s)$는 에피소드를 굉장히 많이 진행해보는 방식으로 얼추 구할 수 있을테니깐 말이다. 책에는 나와있지 않지만 식 {eq}`policy-gradient-theorem`을 다음과 같이도 나타낼 수 있다.

$$
\begin{matrix}
\nabla_{\theta} J(\theta) & \propto & \sum_s d_{\pi_{\theta}}(s) \sum_{a} Q^{\pi_{\theta}}(s,a) \nabla_{\theta} \pi_{\theta}(a|s) & \\
& = & \sum_s d_{\pi_{\theta}}(s) \sum_{a} \pi(a|s) Q^{\pi_{\theta}}(s,a) \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}  & \quad (a) \\
& = & \sum_s d_{\pi_{\theta}}(s) \sum_{a} \pi(a|s) Q^{\pi_{\theta}}(s,a) \nabla_{\theta} \log \pi_{\theta}(a|s)  & \quad (b) \\
& = & \mathbb{E}_{\pi_{\theta}} \left[  Q^{\pi_{\theta}}(S_t, A_t) \nabla_{\theta} \log \pi_{\theta}(A_t|S_t) \right]  & \quad (c) \\
\end{matrix}
$$

<br>

$(a)$은 그냥 $\frac{\pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}$를 곱해주고 위치만 바꾼 것이다. $(b)$는 $\frac{d}{dx} \log f(x)=\frac{f'(x)}{f(x)}$임을 사용한 것이다. 
마지막으로 $(c)$는 $\mathbb{E}\left[ X \right] = \sum_{x}x\;p(x)$임을 사용한 것인데, 확률 $p(x)$에 해당하는 부분은 $d_{\pi_{\theta}}(s)\pi_{\theta}(a|s)$이고, 확률변수 $X$에 해당하는 부분이 $Q^{\pi_{\theta}}(S_t,A_t) \nabla \log \pi_{\theta}(A_t|S_t)$이다. 확률변수 (random variable)은 대문자, 결과 (outcome)은 소문자로 표기해주었다. 식 $(c)$처럼 적어주면 좋은 이유는, 실제 기댓값은 구하기 어렵겠지만, 에피소드를 많이 반복하여  $Q^{\pi_{\theta}}(s,a) \nabla \log \pi_{\theta}(a|s)$를 얻고 표본 평균을 내어 실제 기댓값에 근사할 수 있다는 것이다. 그리고 식 $(c)으$로 보는 것이 이후 REINFORCE나 Actor-Crtic 알고리즘을 설명할 때 더 용이하다. 

<br>

---

## 증명

증명의 편의성을 위하여 유한 상태 공간 및 유한 행동 공간임을 가정하자. 연속일 경우 summation을 적분으로 바꿔주면 된다. 먼저, 우리의 목적함수는 다음과 같다.

$$
J(\theta) := \mathbb{E}_{S_0 \sim d_0}\left[ V^{\pi_\theta}(S_0) \right]=\sum_{s_0 \in \mathcal{S}}d_0(s_0)V^{\pi_\theta}(s_0).
$$

<br>

목적함수를 최대화하기 위하여 우리는 gradient ascent를 사용할 것이며, gradient ascent를 위해서는 목적함수의 그레디언트가 계산해야 한다. 양변에 그레디언트를 취해보자.

$$
\nabla_\theta J(\theta) =  \nabla_{\theta} \sum_{s_0 \in \mathcal{S}}d_0(s_0)V^{\pi_\theta}(s_0) = \sum_{s_0 \in \mathcal{S}}d_0(s_0) \nabla_{\theta}V^{\pi_\theta}(s_0).
$$

<br>

위 식에서 $d_0$에는 $\theta$가 없어서 상수 취급되고, $\theta$에 종속적인 $V^{\pi_\theta}(s_0)$만 그레디언트를 취해준 것이다. 먼저, 상태가치함수는 행동가치함수의 기댓값이라는 성질을 이용하자. 즉, 다음과 같은 등식이 모든 $s \in \mathcal{S}$, $a \in \mathcal{A}$에 대해 성립한다.

$$
V^{\pi} (s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^{\pi}(s, a).
$$

<br>

위 성질을 $V^{\pi_\theta}(s_0)$에 대입하자.

$$
\nabla_\theta J(\theta) =  \ \sum_{s_0 \in \mathcal{S}}d_0(s_0) \nabla_{\theta}\left( \sum_{a_0 \in \mathcal{A}} \pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0) \right).
$$

<br>

위 식에서 $\theta$ 에 종속적인 부분은 $\pi_\theta(a_0|s_a)Q^{\pi_\theta}(s_0, a_0)$이다. 미분의 곱셈 법칙을 사용하자.

$$
\nabla_\theta J(\theta) =  \ \sum_{s_0 \in \mathcal{S}}d_0(s_0) \sum_{a_0 \in \mathcal{A}} \left( \nabla_\theta\pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0) + \pi_\theta(a_0|s_0) \nabla_\theta Q^{\pi_\theta}(s_0, a_0) \right).
$$

<br>

이제 행동가치함수의 재귀적 성질을 이용하자. 즉, 다음 성질을 이용할 것이다.

$$
Q^{\pi}(s, a) = r + \gamma \sum_{s'}p(s' | s, a) V^{\pi}(s'), \text{ where } r=r(s,a).
$$

<br>

위 성질을 $Q^{\pi_\theta}(s_0, a_0)$에 대입하자.

$$
\nabla_\theta J(\theta) =  \ \sum_{s_0 \in \mathcal{S}}d_0(s_0) \sum_{a_0 \in \mathcal{A}} \left( \nabla_\theta\pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0) + \pi_\theta(a_0|s_0) \nabla_\theta \left( r_0 + \gamma \sum_{s_1 \in \mathcal{S}}p(s_1 | s_0, a_0) V^{\pi_\theta}(s_1)  \right)  \right).
$$

<br>

$r_0=r(s_0, a_0)$은 보상함수로부터 계산되기 때문에 $\theta$에 종속적이지 않다. 따라서 그레디언트가 취해지면 0이 된다. $V^{\pi_\theta}(s_1)$는 $\theta$에 종속적이기 때문에 그레디언트를 취해줘야 한다. 즉, 다음과 같이 정리된다.

$$
\nabla_\theta J(\theta) =   \sum_{s_0 \in \mathcal{S}}d_0(s_0) \sum_{a_0 \in \mathcal{A}} \left( \nabla_\theta \pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0) + \gamma  \pi_\theta(a_0|s_0) \sum_{s_1 \in \mathcal{S}} p(s_1 | s_0, a_0) \nabla_\theta V^{\pi_\theta}(s_1)\right).
$$

<br>

잠시 짚고 넘어가자면, 우리는 처음에 $\nabla_\theta V^{\pi_\theta}(s_0)$부터 시작해서 가치함수의 성질을 이용하여 위 식까지 도달한 것이다. 즉,

$$
\nabla_\theta V^{\pi_\theta}(s_0)=\sum_{a_0 \in \mathcal{A}} \left( \nabla_\theta \pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0) + \gamma  \pi_\theta(a_0|s_0) \sum_{s_1 \in \mathcal{S}} p(s_1 | s_0, a_0) \nabla_\theta V^{\pi_\theta}(s_1)\right),
$$

으로 전개해주었다.

<br>

한편, 지금까지의 $\nabla_\theta J(\theta)$ 식에는 크게 두 항이 있다.

$$
\sum_{s_0 \in \mathcal{S}}d_0(s_0)\sum_{a_0 \in \mathcal{A}} \nabla_\theta \pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0),
$$

$$
\gamma \sum_{s_0 \in \mathcal{S}}d_0(s_0) \sum_{a_0 \in \mathcal{A}}   \pi_\theta(a_0|s_0) \sum_{s_1 \in \mathcal{S}} p(s_1 | s_0, a_0) \nabla_\theta V^{\pi_\theta}(s_1).
$$

<br>

두 번째 항에 $\nabla_\theta V^{\pi_\theta}(s_1)$이 있다. $\nabla_\theta V^{\pi_\theta}(s_0)$에 대해 했던 것을 그대로 해주면 다음과 같을 것이다.

$$
\nabla_\theta V^{\pi_\theta}(s_1)=\sum_{a_1 \in \mathcal{A}} \left( \nabla_\theta \pi_\theta(a_1|s_1)Q^{\pi_\theta}(s_1, a_1) + \gamma  \pi_\theta(a_1|s_1) \sum_{s_2 \in \mathcal{S}} p(s_2 | s_1, a_1) \nabla_\theta V^{\pi_\theta}(s_2)\right),
$$

<br>

아래 첨자를 1씩 증가시켜준 것이다. 이를 $\nabla_\theta V^{\pi_\theta}(s_1)$에 대입해주자.

$$
\begin{align*}
\nabla_\theta
 J(\theta) & = & \sum_{s_0 \in \mathcal{S}}d_0(s_0)  \sum_{a_0 \in \mathcal{A}} \Bigg( \nabla_\theta \pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0) + \gamma  \pi_\theta(a_0|s_0) \sum_{s_1 \in \mathcal{S}} p(s_1 | s_0, a_0)  \bigg( \\&&
 \sum_{a_1 \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a_1|s_1)Q^{\pi_\theta}(s_1, a_1) + \gamma  \pi_\theta(a_1|s_1) \sum_{s_2 \in \mathcal{S}} p(s_2 | s_1, a_1) \nabla_\theta V^{\pi_\theta}(s_2) \Big) \bigg) \Bigg).
\end{align*}
$$

<br>

그리고 위 식에는 총 3항이 있다.

$$
\sum_{s_0 \in \mathcal{S}}d_0(s_0)\sum_{a_0 \in \mathcal{A}} \nabla_\theta \pi_\theta(a_0|s_0)Q^{\pi_\theta}(s_0, a_0),
$$

$$
\gamma \sum_{s_0 \in \mathcal{S}}d_0(s_0)\sum_{a_0 \in \mathcal{A}} \pi_\theta(a_0|s_0) \sum_{s_1 \in \mathcal{S}} p(s_1 | s_0, a_0) \sum_{a_1 \in \mathcal{A}}\nabla_\theta \pi_\theta(a_1|s_1)Q^{\pi_\theta}(s_1, a_1),
$$

$$
\gamma^2 \sum_{s_0 \in \mathcal{S}}d_0(s_0)\sum_{a_0 \in \mathcal{A}} \pi_\theta(a_0|s_0) \sum_{s_1 \in \mathcal{S}} p(s_1 | s_0, a_0) \sum_{a_1 \in \mathcal{A}}\pi_\theta(a_1|s_1)\sum_{s_2 \in \mathcal{S}} p(s_2 | s_1, a_1) \nabla_\theta V^{\pi_\theta}(s_2) ,
$$

<br>

규칙성이 잘 보일지 모르겠다. $\nabla_\theta V^{\pi_\theta}(s_t)$를 한번 전개 할때마다, 상태 $s_t$까지 도달할 확률을 곱해주고 각 상태 $a_t$에 대한 $\nabla_\theta \pi_\theta(a_t|s_t) Q^{\pi_\theta}(s_t, a_t)$와 $\nabla_\theta V^{\pi_\theta}(s_{t+1})$ 항이 추가된다. 그리고 후자의 경우 다시 같은 원리로 전개할 수 있다. 이 전개 과정을 무한히 많이 수행한다고 하면  $\sum\limits_{a \in \mathcal{A}}\nabla_\theta \pi_\theta(a|s)Q^{\pi_\theta}(s, a)$ 텀이 무한히 많아질 것이다. 이를 적어보면 다음과 같다.

$$
\nabla_\theta J(\theta) = \sum_{t=0}^{\infty} \gamma^{t} \text{Pr}(s_t =s|\pi_\theta) \sum_{a \in \mathcal{A}} Q^{\pi_\theta}(s,a)\nabla_\theta \pi_\theta(a|s) ,
$$

<br>

여기서 $\text{Pr}(s_t=s | \pi_\theta)$은 정책 $\pi_\theta$를 따랐을 때 $t$ 시점에서의 상태가 $s$일 확률이다. 위 식을 깔끔하게 기댓값 표현으로 나타내고 싶다. 만약 각 상태 $s$에서 다음과 같은 함수를 정의하면 probability distribution일까?

$$
d_\pi(s):= \sum_{t=0}^{\infty} \gamma^t \text{Pr}(s_t=s | \pi_\theta),
$$

<br>

아쉽게도 아니다. 모든 $s$ 에 대해서 $d_{\pi_\theta}(s)$를 더해보면 다음과 같다.

$$
\sum_{s \in \mathcal{S}} \sum_{t=0}^{\infty} \gamma^t \text{Pr} (s_t = s | \pi_\theta) = 
\sum_{t=0}^{\infty} \gamma^t
\sum_{s \in \mathcal{S}}  \text{Pr} (s_t = s | \pi_\theta) = \sum_{t=0}^{\infty} \gamma^t =\frac{1}{1-\gamma}
$$

<br>

1이 되지 않는다. 그래서 위 함수를 보통 unnormalized discounted visited frequencies라고 부른다. 뭐 $\frac{1}{1-\gamma}$로 나눠 주면 probability distribution이 될 것이다. 그래서 확률 분포 $d_{\pi_\theta}$ 를 다시 정의해주자 (사실 식의 생김새는 중요하지 않다.)

$$
d_\pi(s):= (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \text{Pr}(s_t=s | \pi_\theta),
$$

<br>

이를 사용하여 목적 함수의 그레디언트를 다시 적어주면 다음과 같다. 등호가 비례로 바뀌게 된다.

$$
\nabla_\theta J(\theta) \propto \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s)  \sum_{a \in \mathcal{A}} Q^{\pi_\theta}(s,a)\nabla_\theta \pi_\theta(a|s) .
$$

<br>

여기서 로그 함수의 미분 공식과 합성 함수의 미분 공식을 사용하면 위 식을 더 깔끔하게 바꿀 수 있다. 우리는 $(\log f(x))' = \frac{f'(x)}{f(x)}$을 사용할 것이다.

$$
\begin{align*}
\nabla_{\theta} J(\theta) & \propto & \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s)  \sum_{a \in \mathcal{A}} Q^{\pi_\theta}(s,a)  \nabla_\theta \pi_\theta(a|s) \\
& = & \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s)  \sum_{a \in \mathcal{A}} Q^{\pi_\theta}(s,a) \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}  \\
& = & \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s)  \sum_{a \in \mathcal{A}} \pi_\theta(a|s)  Q^{\pi_\theta}(s,a)\nabla_\theta \log \pi_\theta(a|s) \\
& = &\mathbb{E}_{\pi_\theta} \left[ Q^{\pi_\theta}(s,a)\nabla_\theta \log \pi_\theta(a|s) \right],
\end{align*}
$$

<br>

이때, $\mathbb{E}_{\pi_\theta}$는 정책 $\pi_\theta$를 따랐을 때 얻게 되는 $(s, a)$의 확률에 대한 기댓값을 의미한다. 이것으로 증명을 마친다.
