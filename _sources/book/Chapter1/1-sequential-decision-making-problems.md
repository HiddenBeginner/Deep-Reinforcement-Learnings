# 순차적 의사 결정 문제, 에이전트, 환경

강화학습은 순차적 의사 결정 문제 (sequential decision-making problem)을 해결하기 위한 방법론이다. 순차적 의사 결정 문제는 그 이름에서 알 수 있는 것처럼 순차적으로 의사 결정을 하는 문제이다. 순차적 의사 결정 문제를 수행하는 주인공을 에이전트 (agent)라고 부른다. 에이전트가 풀어내고자 하는 또는 통제하고자 하는 대상을 환경 (environment)이라고 부른다. 

순차적 의사 결정 문제에서 에이전트는 매 시점 환경의 상태 (state)를 관측하고 의사 결정을 하여 행동 (action)을 취한다. 
행동을 받은 환경은 상태가 바뀌게 되고 에이전트에게 보상 (reward)을 부여한다. 
그럼 에이전트는 환경의 바뀐 다음 상태를 관측하고 또 다시 행동을 취하며, 환경은 상태가 바뀌고 에이전트에게 보상을 부여한다. 
이 과정을 정해진 횟수나 조건만큼 반복하거나 또는 무한히 반복하는 것을 순차적 의사 결정 문제라고 한다.

에이전트의 목표는 이 과정을 반복하여 얻게 되는 보상들의 총합을 최대화하는 것이다. 에이전트가 환경의 상태에 적절하지 못한 행동을 취했다면 보상을 적게 받았을테고, 환경의 상태에 딱 좋은 행동을 취했다면 보상을 많이 받았을 것이다.


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
