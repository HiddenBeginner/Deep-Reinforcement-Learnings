# Policy Gradient Theorem

Coming soon!

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

규칙성이 잘 보일지 모르겠다. $\nabla_\theta V^{\pi_\theta}(s)$를 한번 전개 할때마다, 상태 $s_t$까지 도달할 확률을 곱해주고 $\sum\limits_{a \in \mathcal{A}}\nabla_\theta \pi_\theta(a|s)Q^{\pi_\theta}(s, a)$과 $\nabla_\theta V^{\pi_\theta}(s')$ 항이 추가된다. 그리고 후자의 경우 다시 같은 원리로 전개할 수 있다. 이 전개 과정을 무한히 많이 수행한다고 하면  $\sum\limits_{a \in \mathcal{A}}\nabla_\theta \pi_\theta(a|s)Q^{\pi_\theta}(s, a)$ 텀이 무한히 많아질 것이다. 이를 적어보면 다음과 같다.

$$
\nabla_\theta J(\theta) = \sum_{t=0}^{\infty} \gamma^{t} \text{Pr}(s_t =s|\pi_\theta) \sum_{a \in \mathcal{A}} Q^{\pi_\theta}(s,a)\nabla_\theta \pi_\theta(a|s) ,
$$

<br>

여기서 $\text{Pr}(s_t=s | \pi_\theta)$은 정책 $\pi_\theta$를 따랐을 때 $t$ 시점에서의 상태가 $s$ 일 확률이다. 위 식을 깔끔하게 기댓값 표현으로 나타내고 싶다. 만약 각 상태 $s$에서 다음과 같은 함수를 정의하면 probability distribution일까?

$$
d_\pi(s):= \sum_{t=0}^{\infty} \gamma^t \text{Pr}(s_t=s | \pi_\theta),
$$

<br>

아쉽게도 아니다. 모든 $s$ 에 대해서 $d_{\pi_\theta}(s)$를 더해보면 다음과 같다.

$$
\sum_{s \in \mathcal{s}} \sum_{t=0}^{\infty} \gamma^t \text{Pr} (s_t = s | \pi_\theta) = 
\sum_{t=0}^{\infty} \gamma^t
\sum_{s \in \mathcal{s}}  \text{Pr} (s_t = s | \pi_\theta) = \sum_{t=0}^{\infty} \gamma^t =\frac{1}{1-\gamma}
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
