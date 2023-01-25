# 가치 함수 근사하기: Stochastic approximation

[Sthochastic approximation](https://en.wikipedia.org/wiki/Stochastic_approximation)은 강화학습 용어는 아니고, 기댓값의 형태로 나타나는 함수 $f(\theta)=\mathbb{E} \left[ F(\theta, \xi ) \right]$를 반복적으로 (iteratively) 근사시키는 방법론이다. 강화학습에서는 가치 함수를 추정할 때 stochastic approximation을 사용한다. 

<br>

Stochastic approximation이 실제 기댓값으로 수렴한다는 것을 증명하는 것은 무진장 어렵다. 미분 방정식과 대학원 수준의 확률론 (확률과 통계 아님)이 필요하다. 필자는 아직 확률론을 수강하지 못했기 때문에 stochastic approximation의 수렴성에 대한 증명을 본 포스팅에 포함시키지 못했다. 하지만, 아이디어는 정말 간단하고 쉽다. 그리고 이번 장에서 나온 업데이트 식이 심층 강화 학습에도 계속 사용되기 때문에 유심히 봐두면 좋을 것이다.

<br>

---

## Stochastic Approximation

어떤 확률 변수 $X$의 기댓값 $\mathbb{E}\left[ X \right]$을 알 수 없을 때, 우리는 주로 표본 평균을 이용하여 실제 평균을 추정한다.

$$
\mathbb{E} \left[ X \right] \approx \frac{1}{N}\sum\limits_{i=1}^{N} X_i,
$$

<br>

이때, $X_i$는 우리의 관측 데이터, $N$은 데이터 개수이다. 데이터의 개수가 굉장히 많을 때 모든 $X_i$를 저장하고 있는 것은 비효율적일 수 있다. 특히, 데이터가 추가될 때마다 평균을 구하는 상황에서는 기존 데이터의 덧셈 계산이 중복되기 때문에 위의 방식으로 표본 평균을 계산하는 것은 비효율적이다. 위 식에서 $X_N$만 시그마 밖으로 빼내서 식을 조작해보자.

$$
\begin{matrix}
S_N  & = & \frac{1}{N} \sum\limits_{i=1}^{N} X_i \\
& = & \frac{1}{N} \sum\limits_{i=1}^{N-1} X_i + \frac{1}{N}X_N \\
& = & \frac{N-1}{N} \frac{1}{N-1} \sum\limits_{i=1}^{N-1} X_i + \frac{1}{N}X_N \\
&=&(1-\frac{1}{N})S_{N-1} + \frac{1}{N}X_N \\
& = & S_{N-1} + \frac{1}{N}(X_N-S_{N-1}).
\end{matrix}
$$

<br>

위 식은 데이터 $X_N$이 추가되었을 때, 표본 평균 $S_N$을 완전히 다시 계산할 필요 없이 현재 평균 $S_{N-1}$과 $X_N$ 그리고 $N$을 통해 계산할 수 있다는 것을 보여준다. 이처럼 표본 평균을 구하는 방법을 **incremental mean**이라고 부른다. 어떻게 보면 $X_i$를 샘플링하면서 점점 $\mathbb{E} \left[ X \right]$에 근사시키는 관점에서 **stochastic approximation**으로 부르기도 한다.

<br>

다음 관계를 유심히 기억하면 Monte Carlo와 TD(0)는 물론 TD(1), TD(2), 모두 유도해낼 수 있다.

$$
\mathbb{E} \left[ X \right] \approx S_{N}=S_{N-1} + \frac{1}{N}\left( X_{N} - S_{N-1} \right).
$$ (incremental_mean)

<br>

식 {eq}`incremental_mean`에는 크게 세 가지 요소가 있다. 

- $X$    : **확률 변수 (random variable)** 
- $X_N$ : 샘플링 또는 관측을 통해 실제 값으로 나타난 확률 변수 $X$의 **실현값 (realization)** 또는 **관측값 (observation)** 이라고도 부름.
- $S_N$  : 기댓값에 대한 **추정값 (estimate**)

강화학습을 공부할 땐, 항상 수식에서 확률 변수와 실현값을 잘 구별할 수 있어야 한다.

<br>

---

## Monte Carlo Evaluation

정책 $\pi$에 대한 상태 $s$의 상태 가치 함수가 다음과 같이 정의된다.

$$
V^{\pi}(s)=\mathbb{E} \left[ G_t |S_t =s \right].
$$

<br>

식 {eq}`incremental_mean`에 그대로 적용해보자. 

- $X$    :  기댓값 안에 있는 $G_t$는 확률 변수 $X$에 해당한다.
- $X_N$ :  $G_t$의 관측값은 $i$ 번째 에피소드에서 상태 $s$에서의 return 값인 $G_{t}^{(i)}$이다.
- $S_N$  :  이전 상태 가치 함수 추정값 $V_{N-1}(s)$은 $S_{N-1}$에 해당한다.

<br>

이를 식 {eq}`incremental_mean`에 그대로 대체해서 적어보면 다음과 같이 Monte Carlo evaluation 업데이트 식이 나온다.

$$
V_{N}(s) \leftarrow V_{N-1}(s) + \frac{1}{N}\left( G_{t}^{(i)} - V_{N-1}(s)\right).
$$

<br>

위 식은 업데이트식이기 때문에 보통 $N$을 제외하고 적어준다. 또한 $\frac{1}{N}$대신 점점 작아지는 작은 값 $\alpha_{N}$을 적는 경우가 많다. 점점 작아지는 상수이어야 수렴성이 증명되지만, 구현에서는 크냥 충분히 작은 상수 하나로 고정하여 사용해도 된다.

$$
V(s) \leftarrow V(s) + \alpha\left( G_{t}^{(i)} - V(s)\right).
$$

<br>

---

## Temporal Difference Evaluation

이전 장의 식 {eq}`1step_state_value_function`의 $(**)$를 다시 적어주면 다음과 같다.

$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left[ R_t + \gamma V^{\pi}(S_{t+1}) | S_t = s \right].
$$


<br>    

무튼 위 식도 결국 기댓값의 형태로 표현되어 있기 때문에 stochastic approximation을 사용할 수 있다. 

- $X$    :  기댓값 안에 있는 $R_{t+1} + \gamma V^{\pi}(S_{t+1})$이 확률 변수 $X$에 대응한다.
- $X_{N}$ :  $R_{t} + \gamma V^{\pi}(S_{t+1})$의 관측값은 $r_t + \gamma V_{k-1}(s_{t+1})$이다.
- $S_N$  :  이전 상태 가치 함수 추정값 $V_{N-1}(s_t)$ 

<br>

이를 식 {eq}`incremental_mean`에 대체해서 적어보면 다음과 같이 TD learning의 업데이트 식이 나온다.

$$
V_N(s_t) \leftarrow  V_{N-1}(s_t) + \frac{1}{N} \left( \left[ r_{t+1} + \gamma V_{N-1}(s_{t+1}) \right] - V_{N-1}(s_t) \right).
$$

<br>

Monte Carlo 때와 마찬가지로 $N$을 생략해주고, $\frac{1}{N}$을 $\alpha$로 적어주면 다음과 같아진다.

$$
V(s_t) \leftarrow V(s_t) + \alpha \left( \left[ r_{t+1} + \gamma V(s_{t+1}) \right] - V(s_t) \right).
$$


```{raw} html
<script
   type="text/javascript"
   src="https://utteranc.es/client.js"
   async="async"
   repo="HiddenBeginner/Deep-Reinforcement-Learnings"
   issue-term="pathname"
   theme="github-light"
   label="💬 comment"
   crossorigin="anonymous"
/>
```