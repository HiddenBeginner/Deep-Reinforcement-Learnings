# 심층강화학습 (Deep Reinforcement Learnings)

강화학습이 무엇인지 몰랐을 때, 필자는 강화학습이 그저 딥러닝의 한 분야인줄 알았다.
컴퓨터 비전, 자연어 처리, GNN 분야를 한 번씩 경험해보았으니 강화학습도 쉽게 이해할 수 있을 줄 알았다.
역시 새로운 분야를 공부할 때는 그 분야의 SOTA [^SOTA] 논문을 읽어야 하는 법.
그래서 펼친 Soft Actor-Critic 논문 {cite:p}`haarnoja_soft_2018`. 단 한 문장도 이해하지 못하고 논문을 덮었다.
지금까지 공부해온 딥러닝 분야는 입력 데이터의 형태와 네트워크 구조가 어떤지만 이해하면 됐었는데 강화학습 분야는 그렇지 않았다.

<br>

강화학습은 딥러닝의 한 분야가 아니다.
학문적으로 오랜 시간 동안 연구되어 온 **강화학습**이란 분야에 뉴럴 네트워크를 갖다 쓰는 **심층강화학습**의 성공적인 사례가 보고되면서, 강화학습 분야가 딥러닝 분야처럼 느껴진 것 뿐이다.
딥러닝을 잘 이해하고 있다고 하더라도 심층강화학습 분야를 바로 해볼 수 있는 것이 아니다.
먼저 강화학습 분야를 이해해야만 한다.

<br>

강화학습 분야를 처음부터 꼼꼼하게 공부하면 더할 나위 없이 좋겠지만, 
분명 전통적인 강화학습 내용 중에서 현대의 심층강화학습을 공부하는 데 꼭 필요하지 않은 내용들도 더러 있다. 
이 책의 목표는 딥러닝을 잘 이해하고 있는 분들이 심층강화학습 분야를 빠르게 입문할 수 있도록 도와주는 것이다.

<br>

이 책은 출판을 목적으로 작성된 것이 아니기 때문에 독자의 원활한 이해를 위해서라면 용어를 번역하지 않고 영문 그대로 사용한다거나, 최신 인터넷 밈을 사용하는 등 아주 케주얼하게 내용을 전개해 나갈 것이다. 책을 읽다가 오류를 발견하거나 질문이 생기면 페이지마다 댓글을 남겨주면 감사드리겠다 (문어체와 예의차림 어딘가). 댓글은 깃헙 아이디만 있으면 쉽게 남길 수 있으니 주저하지 말고 남겨주시면 감사드리겠다.

<br>

부디 나의 설명이 여러분들에게 작은 도움이 되었으면 좋겠다.

<br>

<재야의 숨은 초보>

[^SOTA]: State of the art의 줄임말. 해당 분야에서 성능이 제일 좋은 논문을 의미한다.