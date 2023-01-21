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
$$ (1step-return)

<br>

$G_t$를 다음 스탭 return인 $G_{t+1}$으로 표현된 모습이다. 

<br>

---

## 상태 가치 함수의 재귀적 표현

식 {eq}`1step-return`

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
