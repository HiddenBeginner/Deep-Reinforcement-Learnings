# 벨만 방정식: 가치 함수의 재귀적 성질

지난 장에서는 강화학습 분야에서 가장 중요한 개념 중 하나인 가치 함수에 대해 알아보았다. 어떤 정책의 가치 함수는 해당 정책을 따랐을 때 얻게 되는 보상들의 할인된 누적 합의 기댓값으로서, 정책의 좋고 나쁨을 수치로 표현해준다. 이번 장에서는 가치 함수의 중요한 성질에 대해서 알아본다.

## Return의 재귀적 표현

Return의 정의를 다시 한번 적어보자.

$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots,$$

<br>

이때, $R_{t'} = r(S_{t'}, A_{t'}) \; \forall \; t'=t, t+1, \ldots$으로 계산된다. 두 번째 텀부터 $\gamma$를 공통적으로 갖고 있기 때문에 $\gamma$로 묶어주자.

$$
   \begin{matrix}
   G_t & = & R_t + \gamma \left( R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} +\ldots  \right) \\
   & = & R_{t} + \gamma G_{t+1}.
   \end{matrix}
$$ (1step_return)

<br>

$G_t$를 다음 스탭 return인 $G_{t+1}$으로 표현된 모습이다. $G_t$의 다른 형태인 $R_{t} + \gamma G_{t+1}$을 1step return이라고 부른다. 물론, 2step return을 포함하여 $n$-step return도 있지만, 이에 대해서는 나중에 자세히 알아볼 예정이다. 우선은 1step return을 사용해서 가치 함수의 아주 좋은 성질을 유도해보자.

<br>

---

## 상태 가치 함수의 재귀적 표현

식 {eq}`1step_return`을 상태 가치 함수 정의 {eq}`state_value_function`에 대입해보자.

$$
\begin{matrix}
   V^{\pi}(s) & := & \mathbb{E}_{\pi} \left[ G_t | S_t = s \right] & \\
   & = & \mathbb{E}_{\pi} \left[ R_t + \gamma G_{t+1}| S_t = s \right] &  \\
   & = & \mathbb{E}_{\pi} \left[ R_t + \gamma \mathbb{E}_{\pi} \left[ G_{t+1} | S_{t+1}\right] | S_t = s \right] & \quad (*) \\
   & = & \mathbb{E}_{\pi} \left[ R_t + \gamma V^{\pi}(S_{t+1}) | S_t = s \right]. & \quad (**)
\end{matrix}
$$ (1step_state_value_function)

식 {eq}`1step_state_value_function`의 $(*)$은 [Law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation)을 사용한 것이다. 이 법칙에서 중요한 점은 안쪽 기댓값의 조건부에 있는 $S_{t+1}$은 확률 변수 (random variable)이라는 것이다. Law of total expectation를 증명하는 것은 굉장히 간단하지만 확률 변수에 대한 이해가 적을 경우 법칙이 잘 와닿지 않을 수 있다. 법칙에 대한 직관적인 예시 하나를 아래 박스에 들어놓았다.

```{admonition} **Law of total expectation**
Law of total expectation은 $\mathbb{E} \left[ X \right] = \mathbb{E} \left[ \mathbb{E} \left[ X | Y \right] \right]$이다. 쉽게 이해하자면, 1학년 학생들의 평균 키 $(X)$는 각 반 $(Y)$ 학생들의 평균 키를 구하고 다시 평균을 내서 구할 수 있다는 것을 나타낸다. 조금 더 어려운 이야기를 해보자면 우변에서 대괄호 안의 기댓값은 $X$에 대한 기댓값이고, 바깥 기댓값은 $Y$에 대한 기댓값이다. 바깥 기댓값이 없다고 생각하면 $\mathbb{E} \left[ X | Y = y \right]$ 등으로 적어줘야 한다. 예를 들어, 1반의 평균 키를 나타낸다. 모든 반에 대한 평균을 내는 것이 바깥 기댓값이다.
```

<br>

식 {eq}`1step_state_value_function`의 $(**)$은 다시 한번 상태 가치 함수의 정의인 식 {eq}`state_value_function`을 사용한 것이다. 정리하자면, 상태 $s$에서의 상태 가치 함수는 정책을 따랐을 때 바로 받게 되는 보상 $R_{t}$ 더하기 이어지는 다음 상태의 상태 가치 함수의 기댓값이다.

<br>

이제 기댓값의 정의를 사용하여 식 {eq}`1step_state_value_function`의 $(**)$을 다시 적어볼 것이다. 표기의 편의성을 위하여 상태 공간과 행동 공간의 크기가 유한하다고 가정하자 (이를 유한 상태 공간, 유한 행동 공간이라 한다). 확률 변수 $X$의 기댓값의 정의는 다음과 같다. 확률 변수 $X$의 모든 실현값 $x$에 대해서 $x$가 발생할 확률과 $x$를 곱해준 것을 모두 더해준 것이다.

$$\mathbb{E}_{X} \left[ X \right] = \sum_{x}x \cdot p(x).$$
 
<br>

이를 바탕으로 식 {eq}`1step_state_value_function`의 $(**)$을 다시 적어보자. 

$$
\begin{matrix}
V^{\pi}(s) & = & \mathbb{E}_{\pi} \left[ R_t + \gamma V^{\pi}(S_{t+1}) | S_t = s \right] \\
& = & \sum\limits_{a \in \mathcal{A}} \pi(a|s) \left( r(s, a) + \gamma \sum\limits_{s' \in \mathcal{S}}p(s'|s, a) \ V^{\pi}(s') \right).
\end{matrix}
$$ (state_bellman_equation)

<br>

식을 이해하기 위한 첫 걸음은 기댓값 안에 확률 변수 (random variable)가 어떤 것이 있는지이다. 눈에 보이는 확률 변수는 $R_t$와 $S_{t+1}$이다. 하지만 $R_t=r(s, A_t)$로 계산되기 때문에 사실상 확률 변수는 $A_t$와 $S_{t+1}$이 있다. 따라서 기댓값의 정의를 적어줄 때 확률 변수 $A_t$가 정책 $\pi(\cdot | s)$를 따르는 것부터 시작하면 된다. 이는 식 {eq}`state_bellman_equation`에서 $\sum\limits_{a \in \mathcal{A}} \pi(a|s)$에 해당한다.

<br>

다음으로 행동이 $A_t=a$라고 할 때, 보상 $R_t$은 $r(s, a)$로 결정된다. 여기까지가 $\sum\limits_{a \in \mathcal{A}} \pi(a|s) \cdot r(s, a)$에 해당한다. 한편, 다음 상태 $S_{t+1}$은 전이 확률 분포 $p(\cdot|s, a)$에서 샘플링된다. 확률 변수 $S_{t+1}$의 각 실현값 $s'$에 대한 확률 계산이 $\sum\limits_{a \in \mathcal{A}} \pi(a|s) \sum\limits_{s' \in \mathcal{S}}p(s'|s, a)$에 해당하고, 여기에 $\gamma V^{\pi}(s')$을 곱해준다.

<br>

식 {eq}`state_bellman_equation`은 상태 가치 함수라면 만족하는 등식이며, 이 식을 상태 가치 함수에 대한 벨만 방정식 Bellman equation이라고 부른다.

<br>

---

## 행동 가치 함수의 재귀적 표현

상태 가치 함수의 벨만 방정식을 유도했던 것처럼 행동 가치 함수도 재귀적으로 나타낼 수 있다. 상태 가치 함수와 다른 부분은 행동 가치 함수 $Q^{\pi}(s, a)$는 확률 변수 $A_t$가 $a$로 고정되었다는 것이다. 아래 식을 보자. 

$$
\begin{matrix}
   Q^{\pi}(s, a) & := & \mathbb{E}_{\pi} \left[ G_t | S_t = s, A_t = a \right] & \\
   & = & \mathbb{E}_{\pi} \left[ R_t + \gamma G_{t+1}| S_t = s, A_t = a \right] &  \\
   & = &  r(s, a)  + \mathbb{E}_{\pi} \left[\gamma G_{t+1}| S_t = s, A_t = a \right] &  \quad (*) \\
\end{matrix}
$$ (1step_action_value_function)

<br>

식 {eq}`1step_action_value_function`의 $(*)$에서 $S_t=s, A_t=a$로 고정되었으니 보상 $R_t=r(s, a)$로 딱 결정된다. $R_t$에 더 이상 확률 변수가 없기 때문에 기댓값 계산에서 상수가 되어 밖으로 나올 수 있게 된다. 이제 Law of total expectation 법칙을 사용할 차례이다. 여기서 2가지 선택권이 있다. 쉬운 버전부터 보자

$$
\begin{matrix}
   Q^{\pi}(s, a) & = &  r(s, a)  + \mathbb{E}_{\pi} \left[\gamma G_{t+1}| S_t = s, A_t = a \right] &  \\
   & = &  r(s, a)  + \mathbb{E}_{\pi} \left[\gamma \mathbb{E} \left[ G_{t+1} | S_{t+1}\right] | S_t = s, A_t = a \right] &  \quad (*) \\
   & = &  r(s, a)  + \gamma \mathbb{E}_{\pi} \left[ V^{\pi}(S_{t+1}) | S_t = s, A_t = a \right] &  \quad(**) \\
   & = & r(s, a) + \gamma \sum\limits_{s' \in \mathcal{S}}p(s'|s, a) \ V^{\pi}(s'). & \quad(***)
\end{matrix}
$$ (action_bellman_equation1)

<br>

식 {eq}`action_bellman_equation1`의 $(*)$는 안쪽 기댓값의 조건부에 확률 변수 $S_{t+1}$만 추가된 형태이다. 식 {eq}`action_bellman_equation1`의 $(**)$은 행동 가치 함수와 상태 가치 함수 사이의 관계를 잘 보여준다. 행동 가치 함수 $Q^{\pi}(s, a)$는 보상 $r(s, a)$에다가 전이 확률 분포를 따라 방문할 다음 상태들의 상태 가치 함수의 기댓값에 $\gamma$를 곱해 더해준 것이다. 

<br>

식 {eq}`action_bellman_equation1`의 $(***)$를 상태 가치 함수에 대한 벨만 방정식 {eq}`state_bellman_equation`에 집어 넣으면 다음과 같은 관계식도 유도할 수 있다.

$$
\begin{matrix}
V^{\pi}(s) & = & \sum\limits_{a \in \mathcal{A}} \pi(a|s) \left( r(s, a) + \gamma \sum\limits_{s' \in \mathcal{S}}p(s'|s, a) \ V^{\pi}(s') \right) \\
& = & \sum\limits_{a \in \mathcal{A}} \pi(a|s) Q^{\pi}(s, a) \\
& = & \mathbb{E}_{a\sim\pi(\cdot|s)} \left[ Q^{\pi}(s, a) \right].
\end{matrix}
$$ (relation_between_state_action_value_function)

<br>

식 {eq}`relation_between_state_action_value_function`는 상태 $s$에서의 상태 가치 함수는 정책을 따라 선택한 행동 $a$의 행동 가치 함수의 기댓값이라는 것을 말해준다. 

<br>

한편, Law of total expectation을 적용할 때 안쪽 기댓값의 조건부에 확률 변수 $A_{t+1}$까지 넣어주면 다음과 같은 식이 유도된다.

$$
\begin{matrix}
   Q^{\pi}(s, a) & = &  r(s, a)  + \mathbb{E}_{\pi} \left[\gamma G_{t+1}| S_t = s, A_t = a \right] &  \\
   & = &  r(s, a)  + \mathbb{E}_{\pi} \left[\gamma \mathbb{E} \left[ G_{t+1} | S_{t+1}, A_{t+1} \right] | S_t = s, A_t = a \right] & \quad (*) \\
   & = &  r(s, a)  + \mathbb{E}_{\pi} \left[\gamma Q^{\pi}(S_{t+1}, A_{t+1}) | S_t = s, A_t = a \right] & \quad (**) \\
   & = & r(s, a) + \gamma \sum\limits_{s' \in \mathcal{S}}p(s'|s, a) \sum\limits_{a' \in \mathcal{A}} \pi(a'|s') Q^{\pi}(s', a') & \quad(***)
\end{matrix}
$$ (action_bellman_equation2)

<br>

식 {eq}`action_bellman_equation2`의 $(*)$는 안쪽 기댓값의 조건부에 확률 변수 $S_{t+1}$와 $A_{t+1}$가 있는 형태이다. 확률 변수를 2개를 넣어주는 것도 가능하다 (가능할껄...?). 식 {eq}`action_bellman_equation2`의 $(***)$은 행동 가치 함수의 재귀적 성질을 잘 보여주며, 이를 행동 가치 함수에 대한 벨만 방정식이라고 부른다.

<br>

---

이번 장에서는 가치 함수의 중요한 성질인 벨만 방정식에 대해 알아보았다. 가치 함수는 정책의 성능을 평가할 수 있는 아주 중요한 지표이다. 하지만 많은 경우 가치 함수를 직접 계산하는 것은 불가능 (intractable)하다. 따라서 우리는 가치 함수를 추정 (estimate)하게 될 것인데, 이때 벨만 방정식이 유용하게 사용된다. 

<br>

미리 스포를 하자면, 실제 가치 함수이면 벨만 방정식을 만족해야 한다는 성질을 이용해서 벨만 방정식의 좌변과 우변의 차이가 0이 되도록 가치 네트워크 $V_{\theta}(s)$나 $Q_{\theta}(s, a)$를 학습시킨다. 이에 대한 근거가 되는 방법론인 stochastic approximation 다음 장에서 알아보자.


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
