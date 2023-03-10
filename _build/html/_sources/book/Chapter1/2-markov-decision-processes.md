# Markov Decision Process (MDP)

우리의 목표는 순차적 의사 결정 문제를 해결하는 것이다. 하지만 아직 순차적 의사 결정 문제는 너무 추상적인 대상이다. 상태는 무엇이고, 행동은 무엇이며, 환경은 어떤 규칙에 의해 상태를 변경하며, 어떤 규칙에 의해 에이전트에 보상을 주는 것일까? 우리는 순차적 의사 결정 문제를 해결하기 전에 먼저 순차적 의사 결정 문제를 수학적으로 기술할 수 있는 틀이 필요하다.

<br>

Markov Decision Process (MDP)는 순차적 의사 결정 문제를 적당히 단순화하여 정의할 수 있게 만들어주는 틀이다. 어떤 순차적 의사 결정 문제를 MDP로 모델링할 경우, 이 “순차적 의사 결정 문제는 MDP를 따른다”라고 말한다. MDP는 여섯 가지를 정의하여 순차적 의사 결정 문제를 기술한다. 여섯 가지는 각각 상태 공간 $\mathcal{S}$, 행동 공간 $\mathcal{A}$, 초기 상태의 확률 분포 $\rho_0$, 보상 함수 $r$, 전이 확률 분포 $p$, 할인률 $\gamma$이다. 조금 더 학술적으로는 한 MDP는 순서쌍 $(\mathcal{S}, \mathcal{A}, r, \rho_0, p, \gamma)$으로 정의된다고 말한다.

<br>

## 상태 공간 (state space)과 행동 공간 (action space)

상태 공간 $\mathcal{S}$는 환경이 가질 수 있는 모든 상태들의 집합이다. 6축 관절 로봇 제어의 경우, 각 관절의 각도로 로봇의 현재 상태를 표현할 수 있기 때문에 상태 공간 $\mathcal{S}$는 $\mathbb{R}^6$의 부분 집합이다. 부분 집합이라고 표현한 이유는 각 관절이 가질 수 있는 최대 각도가 정해져 있기 때문이다. 일부 관절은 $0^\circ$부터 $360^\circ$ 사이의 값만 가질 수 있을 것이고, 일부 관절은 $0^\circ$부터 $180^\circ$ 사이의 값만 가질 수 있을 것이다. 상태는 주로 $s \in \mathcal{S}$로 표기해준다. 특정 시점의 상태 (예를 들어 $t$ 시점의 상태)를 나타내고 싶을 경우 $s_t$으로 표기해준다.

<br>

행동 공간 $\mathcal{A}$는 에이전트가 취할 수 있는 모든 행동들의 집합이다. 6축 관절 로봇 제어의 경우, 각 관절을 몇 도만큼 회전하는 것을 행동으로 정의한다면 행동공간 $\mathcal{A}$는 $\mathbb{R}^6$의 부분 집합이다. 단, 각 관절마다 한번에 회전할 수 있는 최대 각도가 제한되어 있을 것이기 때문에 부분 집합이라고 표현했다. 슈퍼 마리오 게임의 경우 행동은 왼쪽으로 이동, 오른쪽으로 이동, 앉기, 점프가 있을 것이다. 행동은 주로 $a \in \mathcal{A}$로 표기해준다. 특정 시점의 행동 (예를 들어 $t$ 시점의 상태)를 나타내고 싶을 경우 $a_t$으로 표기해준다.

<br>

## 보상 함수 (reward function)
보상 함수 $r$은 에이전트가 특정 상태에서 취한 행동의 좋고 나쁨을 나타내는 함수이다. 논문마다 보상 함수의 정의역이 조금씩 다른데, 가장 보편적인 보상 함수는 상태 $s$에서 행동 $a$에 대해서 보상을 부여한다. 즉, $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ 인 함수이다. 다음으로 많이 사용되는 정의는 상태 $s$에서 행동 $a$를 취하여 상태가 $s'$으로 바뀐 것에 대해서 보상을 부여한다. 즉, $r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ 인 함수이다. 따로 언급하지 않는 이상 이 책에서는 첫 번째 정의를 사용할 것이다.

<br>

보상 함수는 각 상태, 행동 순서쌍 $(s, a)$에 대해서 딱 정해진 (determinitic) 보상을 부여할 수도 있지만, 더 일반적으로 어떤 확률 분포에서 보상을 stochastic하게 샘플링하여 부여할 수도 있다. 추가적으로 강화학습 알고리즘의 수렴성을 보장하기 위해서 보상의 최대값이 정해져있다고 가정한다. 즉, 보상 함수는 bounded 되어 있다.

<br>

## Trajectory와 전이 확률 분포 (transition probability distribution)

초기 상태 확률 분포 $\rho_0$는 말 그대로 환경이 가질 수 있는 초기 상태의 확률 분포이다. 각 상태마다 환경이 해당 상태를 초기 상태로 가질 확률이 정의되어 있는 함수로 해석할 수 있기 때문에 $\rho_0:\mathcal{S} \rightarrow [0,1]$로 적어준다. $\rho_0$는 각 상태를 0과 1사이의 값으로 보내는 함수라고 읽으면 된다. 초기 상태, 즉 첫 번째 시점에서의 환경의 상태를 $s_0 \sim \rho_0$로 적어준다. 환경의 첫 번째 상태 $s_0$는 초기 상태 확률 분포 $\rho_0$에서 샘플링되었다는 의미이다.

<br>

환경의 초기 상태 $s_0$에서 시작해서 에이전트는 행동을 취하기 시작한다. 에이전트가 $s_0$에 대해 취한 행동을 $a_0$, 받은 보상을 $r_0=r(s_0, a_0)$이라고 표기하자. 그리고 다음 시점인 $t=1$에서의 환경의 상태를 $s_1$, 대응하는 에이전트의 행동을 $a_1$, 받은 보상을 $r_1=r(s_1,a_1)$, … 으로 적어주면 우리는 이 일련의 과정의 다음과 같이 적어줄 수 있다. 

$$
\tau=(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T),
$$

여기서 $r_t = r(s_t, a_t)$이고, 이 일련의 과정 $\tau$ (타우라고 읽음)을 trajectory라고 부른다. Trajectory를 직역하면 탄도, 궤도, 궤적인데 의미가 직접적으로 와닿지 않아서 trajectory라고 부를 것이다. 

$t$ 시점에서의 환경의 상태 $s_t$는 초기 상태부터 시작해서 $t-1$번 째 행동까지가 만들어낸 산출물이다. 따라서 $t$ 시점에서 어떤 상태가 발생할지 알고 싶으면 다음과 같은 조건부 확률 분포를 고려해야 한다.

$$
p\left(s_t|s_0, a_0, s_1, a_1, \ldots, s_{t-1},a_{t-1}\right).
$$

<br>

이 조건부 확률이 어떻게 정의되는지 모르겠지만, 우리가 다루고 계산하기 굉장히 어려울 것이 분명하다. 앞전에 MDP가 순차적 의사 결정 문제를 적당히 단순화하여 모델링한다고 언급했었는데, MDP는 $t$ 시점의 상태를 $t-1$ 시점의 상태와 행동에 의해서만 결정된다고 가정하여 문제를 단순화시킨다. 즉, 

$$
p\left(s_t|s_0, a_0, s_1, a_1, \ldots, s_{t-1},a_{t-1}\right)=p\left(s_t|s_{t-1},a_{t-1}\right),
$$

라고 가정을 하는 것이다. 이 가정이 성립할 경우 해당 순차적 의사 결정 문제가 Markov property를 만족한다고 말한다. 순차적 의사 결정 문제가 Markov property를 성립한다고 가정하기 때문에 Markov decision process라고 부르는 것이다.

<br>

처음에는 이 가정이 합리적인 가정인지 잘 와닿지 않는다. 현재의 상태를 기술하기 위해서는 과거의 모든 상태와 행동을 알아야 하는 것이 아닐까 싶은 것이다. 예를 들어 100m 달리기를 생각해보자. 문제 단순화를 위하여 원점에서 속도가 0인 상태로 시작하고 가속도는 일정하다고 가정하자. 상태 공간을 위치와 속도로 정의해보자. $t$ 시점에서 위치와 속도는 과거 내가 어디에 있었는지, 속도는 몇이었는지 전부 다 알 필요 없이 $t-1$ 시점에서의 위치와 속도만 알면 완벽하게 결정된다. 

<br>

Markov property를 가정하여 상태가 어떻게 바뀌는지 더 쉽게 계산할 수 있게 되었다. 전이 확률 분포 (transition probability distribution) $p$는 상태 $s$에서 행동 $a$를 취했을 때 환경의 상태가 $s'$으로 전이할 확률을 나타낸다. 즉, $p: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0, 1]$인 함수이고, 각 함수값마다 확률이 부여되어 있다. 즉, $p(s, a, s') = \text{Pr}[s' | S_t=s, A_t=a]$으로 정의되며, 직관성을 위해 $p(s, a, s')$ 대신 $p(s'|s, a)$으로 표기해준다. 

<br>

## 할인률 (discount factor)
할인률 $\gamma \in [0, 1]$ 은 0과 1사이의 값을 갖는 실수값이며, 더 나중에 받은 보상일수록 더 낮은 가중치를 부여하는 역할을 한다. 예를 들어, trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$가 주어졌을 때, 이 trajectory에서 받은 총 누적 보상을 계산할 때 단순하게 더해주는 대신 다음과 같이 가중합을 하는 것이다.

$$r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \ldots$$

<br>

할인률에 대해서는 다음 절에서 더 자세히 알아볼 예정이다.


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
