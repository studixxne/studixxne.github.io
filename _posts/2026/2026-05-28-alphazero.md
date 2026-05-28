---
title: "AlphaZero를 통해 십이장기 AI 구현하기"
category: RL
tags: [인공지능, RL, PyTorch, Project, 구현]
comment: true
---

### Introduction

딥러닝을 배우고 처음으로 진행하는 프로젝트로 AlphaZero로 보드게임 AI를 만들어 보았다. 결과적으로는 SOTA 모델을 만들어내는 데 성공하였다.  

적용한 Task는 tvn의 서바이벌 예능 프로그램 '더 지니어스'에서 나온 '십이장기'라는 게임인데, 간단하게 설명하자면 4*3로 축소한 장기이다. 자세한 규칙은 [링크](https://namu.wiki/w/%EC%8B%AD%EC%9D%B4%EC%9E%A5%EA%B8%B0)를 참조하자.  

학창 시절에 알파고를 감명깊게 보기도 했고, 데미스 하사비스를 롤모델로 삼으며 강화학습에 큰 관심을 가졌었기에 AlphaZero를 직접 만들어보겠다고 결정했다. 십이장기도 마찬가지로 학창 시절에 몰입하면서 보던 프로그램이 더 지니어스였기 때문에 결과적으로 AlphaZero 십이장기 AI 제작에 대한 동기가 되었다.  



### Why AlphaZero?

십이장기는 방송용 보드게임이기 때문에, 기보가 쌓여있지 않고 전략 연구도 이루어지지 않은 백지 상태(tabula rasa)인 상황이다.

십이장기와 같은 작은 규모의 게임에서는 Alpha-Beta Pruning 같이 효율적인 Tree Search로도 해결할 수 있지만, 문제는 Tree Search를 위해서는 현재 판도를 Evaluation을 통해서 점수를 계산해야 한다는 것이다.  

하지만 십이장기의 상황은 백지 상태이기 때문에 평가 함수를 작성하는 것이 매우 어려웠고, 만들어진 평가 함수가 최적인지 확신할 수 없다. 그러나 AlphaZero는 백지 상태에서도 Self-Play를 통해 초인간적인 능력을 보일 수 있기 때문에 현재 상황에서 적합하다고 판단하였다.  

2026년 현재, AlphaZero보다 더 성능이 뛰어난 아키텍처도 존재한다. 체스의 엔진 스톡피쉬처럼 NNUE를 도입해서 Tree Search와 결합하는 방식으로 효율성과 성능을 올릴 수 있지만, 십이장기와 같이 소규모 Task에서는 AlphaZero만으로도 충분히 압도적인 성능을 뽑아낼 수 있기에 더 상징성 있고 간단한 AlphaZero를 선택하였다.



### AlphaZero Architecture
#### MCTS

제일 먼저 AlphaZero의 핵심 구조인 MCTS에 대해서 알아야 한다. Pure MCTS는 이전에 포스팅한 적이 있으며 [Post](https://studixxne.github.io/rl/2026/02/21/mcts.html)에서 확인할 수 있다.  

Pure MCTS를 요약하면 무작위 샘플링을 통해서 Simulation을 진행하고 얻어 낸 Reward를 바탕으로 트리를 갱신함으로써 어떤 선택이 유망한지 알려주는 방법론이다. 그러나 기존 MCTS에서는 큰 문제가 존재했는데 바로 Simulation이 무작위로 이루어진다는 것이다. 이로 인해 Rollout 횟수를 늘리더라도 매우 넓은 상태 공간에서는 최적의 수를 찾기 어렵다.  

이러한 무작위 Rollout의 단점을 해결하기 위해 AlphaGo, AlphaZero에서 MCTS에 신경망을 도입했다. MCTS에서 무작위로 Rollout을 하는 대신, 현재 국면을 신경망이 살펴보고 '좋아 보이는 수'를 위주로 탐색해 실질적으로 탐색해야 할 상태 공간을 축소시켜 MCTS의 성능을 더욱 더 이끌어낼 수 있다.  

AlphaZero의 MCTS는 다음과 같다.

![MCTS](/post_image/2026/0528/image1.png)

**(1) Select**  
루트 노드에서 시작해서 PUCT을 최대화하는 Action을 선택하면서 하위 노드로 내려간다.

$$
a_{t}=argmax_{a}(Q(s, a) + U(s, a))
$$

$$
U(s, a) = c_{puct}\ *\ P(s, a)\ *\ \frac{\sqrt{\sum_{b}N(s,b)}}{1+N(s,a)}
$$

탐색은 $Q(s,a)$와 $U(s,a)$의 합이 가장 큰 Action을 선택함으로써 이루어진다. 여기서 $Q(s,a)$는 Exploitation, $U(s, a)$는 Exploration을 나타낸다. 가치가 높은 행동을 선택하되, 방문 횟수가 적거나 정책망의 점수가 높은 경우에는 탐색을 더 하도록 유도해준다.  


**(2) Expansion & Evaluate**  
Leaf Node에 도착하게 되면 Expansion을 수행한다. 신경망을 통해 현재 상태의 정책($P$)과 가치($V$)를 얻어내며 새로운 노드를 트리에 확장한다. 

$$
(P(s_L, \cdot),\  V(s_L)) = f_\theta(s_L)
$$


**(3) Backup (Backpropagation)**  
단계 (2)에서 얻어 낸 $V$ 혹은 Terminal Node인 경우 $Reward$를 통해 경로를 따라 돌아가면서 N과 Q의 값을 업데이트한다.  


위의 세 과정들을 지정된 Simulation 횟수만큼 반복하여 Tree를 확장시킨 후, 최종적으로 다음과 같이 Policy를 반환한다.

$$
\mathbf{\pi}∝\mathbf{N}^{\frac{1}{\tau}}
$$


#### Self-Play

![SelfPlay](/post_image/2026/0528/image2.png)  

Self-Play는 실제로 에이전트가 직접 플레이를 하면서 학습 데이터를 축적하는 단계다. 매 턴마다 MCTS를 통해 $\mathbf{\pi}_t$를 얻어내고 이를 바탕으로 $\mathbf{a}_t$를 선택한다.  

이를 반복해 게임이 종료되면 최종 결과인 $z$를 얻게 되며, 지금까지 누적된 모든 턴의 데이터 ($s_t$, $\pi_t$, $z_t$)를 학습용 버퍼에 저장한다.


#### Neural Network Training

![Training](/post_image/2026/0528/image3.png)  

Self-Play 과정에서 얻어낸 ($s_t$, $\pi_t$, $z_t$)를 통해 신경망($f_\theta)$을 학습시킬 수 있으며 loss는 다음과 같다.

$$
\mathit{l} = (z - v)^2 - \pi^{\text{T}} \log p + c \|\theta\|^2
$$

$(z-v)^2$의 항을 통해 **신경망이 예측한 $v$가 실제 대국 결과인 $z$와 가까워지도록** 하며, $\pi^{\text{T}} \log p$의 항을 통해 **신경망이 예측한 정책 $p$가 MCTS를 수행해서 얻어낸 고품질의 정책인 $\pi$와 가까워지도록** Cross-Entropy loss를 구성한다. 마지막으로 L2 정규화 $c\|\theta\|^2$를 포함하여 Overfitting을 방지해준다.


정리하면 트리를 업데이트할 때 무작위 Rollout이 아닌 신경망의 $V$와 $P$의 예측을 기반으로 이루어지며, MCTS로 얻어진 최종 정책인 $\pi$를 통해 Self-Play를 진행한다. Self-Play를 통해 데이터가 누적되면 $P$를 $\pi$에 가깝게, $v$를 $z$에 가깝게 신경망을 학습시켜준다.

결론적으로, 반복되는 Self-Play로 $p$가 지속적으로 $\pi$를 따라잡으려고 하는 구조로 AlphaZero 모델이 재귀적으로 자가발전하게 된다.



### Implementation

전체 코드는 [Github](https://github.com/studixxne/twJanggi-Zero)로 확인할 수 있다!  

전체 구현에 대해서는 다 다루지 않고 특정 부분만 살펴보자.

#### Resnet을 활용한 신경망 구축
당시 논문에서 사용했던 구조인 Resnet 구조로 네트워크를 작성하였다.  

Residual Block들을 모두 거친 후 마지막 레이어에서 $P$와 $V$를 뽑아낼 수 있도록 Dual Head 구조로 구성했으며, 이는 다음과 같이 코드를 구현했다.

``` python
self.policy_head = nn.Sequential(
    nn.Conv2d(hidden_ch, out_channels=2, kernel_size=1, stride=1),
    nn.BatchNorm2d(2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(24, 132)
)

self.value_head = nn.Sequential(
    nn.Conv2d(hidden_ch, out_channels=1, kernel_size=1, stride=1),
    nn.BatchNorm2d(1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(12, hidden_ch),
    nn.ReLU(),
    nn.Linear(hidden_ch, 1),
    nn.Tanh()
)
```

Policy Head의 경우에는 마지막에 Linear 층으로 132 차원으로 변환해주는데, 여기서 132는 모든 Action의 개수이며 output은 각 Action에 대한 Logit이 된다. 이를 사용할 때에는 invalid action의 logit을 -inf로 치환한 후 softmax를 적용하면서 최종 정책 $P$를 얻어낸다.  

Value Head의 경우에는 마지막에 Tanh를 적용해줌으로써 -1~1의 스칼라 값을 얻어내 현재 상태의 가치를 확인할 수 있도록 한다.


#### 1P와 2P의 구분
AlphaZero를 구현할 때 가장 헷갈리고 어려운 부분이다.  

AlphaZero는 하나의 MCTS를 사용하고, 신경망은 항상 자신이 1P일 때를 기준으로 학습하기 때문에 시점을 통일 시켜줘야 한다. 만약 시점을 통일시켜주지 못한다면 스스로 모델이 자살하는 수를 최적이라고 학습하거나, invalid한 수를 두려고 하는 문제가 발생할 수 있다.  

첫 번째로 MCTS로 만들어진 트리를 공통으로 사용하기 위해서는 Negamax 방식을 활용해야 한다. 이는 backup 과정에서 확인할 수 있다.

``` python
# Terminal Node가 아닌 경우에는 Expansion을 진행
if not done:            
    value = -self.expansion(cur, env, search_history)

# Terminal Node인 경우에는 그냥 Reward 가져옴
else:
    value = reward

self.backpropation(cur, value)
```


``` python
def backpropagation(self, node, v):
    if node == self.root:
        return
    
    cur = node
    while True:
        action = cur.action
        cur = cur.parent
        cur.N[action] += 1
        cur.W[action] += v
        cur.Q[action] = cur.W[action] / cur.N[action]
        v *= -1

        if cur == self.root:
            break
```

만약에 1P의 턴에서 $v$를 얻었을 때 현재 상황이 좋은 점수라면, 부모 노드의 턴은 2P가 되기 때문에 해당 노드는 2P에게 악수가 된다. 구현에 따라 언제 $v$를 더해주냐는 다를 수 있지만, 공통적으로는 반드시 **Backpropagation 과정을 거칠 때 부호 반전**을 해주면서 올라가야 한다.

두 번째로 신경망은 1P를 기준으로 학습함으로 **반드시 2P일 때 Flip 과정**이 필요하다는 것이다.  

제일 먼저 Self-Play에서 데이터 ($s_t$, $\pi_t$, $z_t$)를 누적할 때 $z_t$는 1P, 2P에 따라 상대적인 것이기 때문에 역순으로 따라가며 $z_t$의 부호를 맞춰줘야 한다.

``` python
# Episode 결과를 Data에 추가
for h in data[::-1]:
    h.append(z)
    z *= -1
```

이후 State와 Action을 1P 기준으로 바꿔주는 것이 필요하다. 신경망에 State를 넣기 전에 턴이 2P라면 1P와 동일한 형태로 Flip 해주는 작업이 필요하다.  

``` python
def _get_flip_state(self, state, player):
    if player == 1:
        return state
    
    board, taken_piece, king_enter, *remain = state

    flip_board = np.flip(board * -1)
    flip_taken_piece = {-1: taken_piece[1][:],
                        1: taken_piece[-1][:]}
    flip_king_enter = {-1: king_enter[1],
                        1: king_enter[-1]}
    
    return [flip_board, flip_taken_piece, flip_king_enter] + remain
```

그 후 신경망에서 얻어 낸 정책에 대해서도 Flip 해주는 것이 필요하다. 신경망은 1P를 기준으로 결과를 내기 때문에, 2P에게 Policy를 적용하기 위해서는 2P 관점의 Policy로 Flip하는 과정이 필요하다.  

```
def get_flip_policy(self, policy, player):
    if player == 1:
        return policy
    
    return policy[FLIP_ACTION]
```

이 때 FLIP_ACTION의 경우에는 직접 Action을 Flip 시켜서 얻어낸 테이블을 미리 작성해두어 변환하는 방식으로 구현한다.

#### Exploration을 위한 Noise와 Temperature
AlphaZero는 신경망에서 얻어 낸 $P$를 통해 MCTS를 수행하지만, 다양한 데이터를 수집하기 위해서는 반드시 방문하지 않은 경로도 탐색해야 한다. 이를 위해서 AlphaZero에서는 Dirichlet Noise와 Temperature를 도입하였다.

``` python
if alpha > 0:
    valid_actions = env.get_valid_actions()
    valid_actions_indices = np.where(valid_actions == 1)[0]
    noise = np.random.dirichlet([alpha] * np.sum(valid_actions, dtype=np.int8))

    for i, action in enumerate(valid_actions_indices):
        self.root.P[action] = self.root.P[action] * (1-epsilon) + epsilon * noise[i]
```

Noise의 경우 numpy에 탑재된 함수를 통해서 쉽게 적용할 수 있다.

``` python
# pi 반환
if temperature > 1e-3:
    N = self.root.N ** (1.0 / temperature)
    pi = N / np.sum(N)

else:
    pi = np.zeros(132, dtype=np.float32)
    pi[np.argmax(self.root.N)] = 1.0

return pi
```

정책의 경우 $N^{1/\tau}$를 적용해서 pi를 반환하지만, temperature가 일정 선보다 낮아진 경우에는 사실상 $argmax$와 동일함으로 가장 방문 횟수가 많은 action만 확률을 1로 설정하고 나머지는 0으로 둔다.

#### Encoder와 State 변환
신경망에 $S$를 넣기 위해서는 Encoder를 통해 Tensor로 바꿔주는 작업이 필요하다. 그렇다면 $S$를 어떻게 Tensor로 변환할까?  

``` python
def get_tensor(self, state_list):
    _, _, _, turn, _, player = state_list[-1]
    layers = []

    for state in state_list[::-1]:
        s_board, s_taken_piece, s_king_enter, s_turn, s_repeat, _ = self._get_flip_state(state, player)

        # 내 기물의 위치 / 상대 기물의 위치 (10개)            
        target_piece = np.array([1, 2, 3, 4, 5, -1, -2, -3, -4, -5]).reshape(10, 1, 1)
        layers.append((s_board == target_piece).astype(np.float32))

        # 내 포로 개수 / 상대 포로 개수 (6개)
        taken_array = np.array(s_taken_piece[1] + s_taken_piece[-1]).reshape(6, 1, 1)
        layers.append(np.broadcast_to(taken_array, (6, 4, 3)))

        repeat_array = np.zeros((3, 1, 1))
        repeat_array[s_repeat, 0, 0] = 1.0
        layers.append(np.broadcast_to(repeat_array, (3, 4, 3)))

        king_enter_array = np.array((s_king_enter[1], s_king_enter[-1]), dtype=np.float32).reshape(2, 1, 1)
        layers.append(np.broadcast_to(king_enter_array, (2, 4, 3)))

    layers.append(np.zeros((self.M*(self.T-len(state_list)), 4, 3)))

    # 누구의 턴 (레드(선공) 플레이어면 0, 초록(후공) 플레이어면 1)
    layers.append(np.zeros((1, 4, 3)) if player == 1 else np.ones((1, 4, 3)))
    # 총 이동 수
    layers.append(np.full((1, 4, 3), turn / 150.00))
    
    # (1, M*T+L, 4, 3)의 텐서 반환
    x = torch.tensor(np.concatenate(layers, axis=0), dtype=torch.float32).unsqueeze(0)
    return x
```

보드판의 크기가 (4, 3)를 기준으로 Shape를 결정하였으며, 각각 Layer마다 말의 위치를 1로 표시하고, 아닌 경우에는 0으로 표시하도록 했다. 말의 개수와 같은 정수형의 경우에는 모든 칸을 해당 정수로 채울 수 있도록 구현한다.  

하지만 여기서 주의해야할 점이 Turn과 같이 단독적으로 숫자가 매우 큰 경우에는 올바르게 학습이 되지 않음으로 반드시 다른 Layer와 값이 유사하도록 0에서 1로 점차적으로 올라가도록 정규화해주는 작업이 필요하다.  

또한 천일수를 구현하기 위한 Repeat의 경우에는 1회 반복과 2회 반복의 가치 차이는 매우 크기 때문에 0, 1, 2로 표현하는 것이 아닌 따로 Layer를 나누어서 텐서를 구성한다.

마지막으로 AlphaZero는 State History를 포함해 하나의 텐서로 묶어서 연산함으로 모든 History에 위의 작업을 수행해준 후 하나의 텐서로 묶어주도록 한다.



### Performance

Evaluation은 Rollout 횟수 1000의 Pure MCTS를 기준으로 승률을 평가하였다.  

![win_rate](/post_image/2026/0528/image4.png)

약 **28K의 Self-Play**를 통해서 **Pure MCTS를 상대로 승률 100%를 달성**하였고 이에 따라 학습을 종료하였다.  

실제로 플레이해본 결과 AI는 압도적인 성능을 자랑했다. 직접 플레이했을 때는 단 한 판도 이겨보지 못하였으며, 온라인 커뮤니티에서 **실제 유저들과 대결했을 때도 대부분의 사람을 상대로 전승(무승부 제외, 승률 100%)을 기록**했다.

![react1](/post_image/2026/0528/image5.png)

![react2](/post_image/2026/0528/image6.png)

개인적으로 당시 최강자 라인에게 이런 반응을 들었을 때 엄청난 뽕을 느꼈다..  
물론 이기지는 못하고 작은 Task 특성 상 반복수로 인하여 3판 무승부로 게임이 끝났다.  

이를 통해서 AlphaZero 알고리즘이 십이장기와 같은 작은 Task에 대해서도 백지 상태에서 인간을 뛰어넘는 성능을 낼 수 있다는 것을 확인할 수 있었고, 양쪽이 최적의 수를 두면 무승부로 수렴하는 것도 확인할 수 있었다.



### 직면한 문제점
#### 적절한 Evaluation 기준의 부재
이번 프로젝트를 통해 **Evaluation의 중요성**을 뼈저리게 깨닫게 되었다. 단순 방송용 게임이라 기존에 연구된 기준이 없었기 때문에 스스로 평가 척도를 세웠어야 했지만, 첫 프로젝트인 만큼 이를 간과하고 무작정 학습부터 시작했다가 큰 난관에 부딪혔다.

- **Loss만으로는 부족한 성능 측정**  
학습을 진행하면서 단순히 loss가 떨어지는 것만으로는 모델의 실제 성능을 파악하기가 매우 어려웠다. 초기에는 loss가 급격히 떨어지며 훈련되는 것을 확인할 수 있었지만, 모델이 성숙해지는 중반부터는 loss가 특정 부근에서 진동만 했기에 확인이 어려웠다. 이로 인해 loss가 낮더라도 실질적으로는 Undertrained Model일 가능성이 존재했다.

- **Arena 평가의 한계**  
초기에는 Checkpoint 모델과 직접 플레이해서 성능을 평가할 수도 있지만, 일정 Iter 이상부터는 평범한 사람의 실력을 뛰어넘어 정확한 측정이 불가능해졌다. 그래서 **모델끼리 승부하는 Arena**를 구현하고자 했다.  
하지만 십이장기 특성상 '경우의 수'가 매우 적었기 때문에 최적의 수가 1개에 가까웠다. 이러한 문제로 Arena를 진행하더라도 모델은 매번 똑같은 수를 두어 평가를 하기가 매우 어려웠다. $\tau$를 높여서 다양한 수를 유도해보았지만 모델의 실력보다는 '누가 먼저 악수를 두는지'를 겨루는 확률 싸움으로 변질되어버렸다.  
이를 해결하기 위해 승률이 5:5로 비등한 국면의 데이터셋으로 평가하려는 시도를 했지만, 사전 연구가 없기에 정말로 그 국면이 5:5의 판도인지 평가할 수 없어 불가능했다.

- **해결책: Pure MCTS를 Baseline으로 활용**  
현실적으로 채택 가능한 방법으로써 **Pure MCTS를 상대로 승률을 측정**하기로 결정했다. Pure MCTS는 실제 무작위 Simulation을 기반으로 수를 두기 때문에 예측 불가능한 다양한 상황을 연출할 수 있고, 잘 훈련된 모델이라면 동일한 조건의 Pure MCTS가 만들어내는 상황들을 완벽히 대응해야 했기 때문에 **Pure MCTS 상대로 승률 100%를 달성할 때까지 훈련**을 진행했다.

결과적으로 **Iter 700, 약 28K의 Self-Playing** 끝에 만족스러운 Master Model을 선정할 수 있었다.


#### Self-Play의 데이터 편향 문제
AlphaZero에서는 Dirichlet Noise를 통해서 Exploration를 유도하지만 Self-Play로 데이터를 생성한다는 점 때문에 데이터 편향 문제가 발생할 수 있다.  

- **특수한 상황에서의 편향**  
십이장기에서는 상대의 왕을 잡는 것 외에도 자신의 왕이 상대 진영에서 1턴을 버티게 되면 승리하는 조건이 추가로 존재한다. 상대의 왕을 잡는 것보다 자신의 왕을 상대 진영에 안착시키고 버티는 것이 훨씬 더 어렵다. 만약 이런 승리 조건이 발생하게 된다면 이미 상대의 수비력이 약화된 상태일 것이다. 이런 이유로 실질적인 위협을 인지하지 못하고 **진영에 들어가면 대부분 승리한다는 데이터로만 누적**이 되었을 것이다. 이런 이유로 특정 상황에서는 Undertrained 문제가 발생하였다.

- **Undertrained Model 예시**  
예를 들어 왕이 승리를 위해 상대 진영으로 돌진하기 좋은 위치에 존재할 경우, 신경망은 직관에 따라 value를 매우 높게 평가하며 돌진하려고 하지만 실질적으로는 왕이 공격 당하면서 필패수가 되어버리는 상황이 발생하게 되는 것이다.  
![move1](/post_image/2026/0528/image7.png)  
![move2](/post_image/2026/0528/image8.png)  
![move3](/post_image/2026/0528/image9.png)  
![move4](/post_image/2026/0528/image10.png)  
![log](/post_image/2026/0528/image11.png)  

  *19번째 수에서 최적은 (2, a) to (1, a)이고 (2, a) to (3, a)는 필패수지만 모델을 이를 예측하지 못했고, 상대가 실제로 반격수를 두자 뒤늦게 필패를 예측했다*

- **해결 방안?**  
물론 이런 문제는 MCTS의 Simulation 횟수를 늘려 신경망의 오류를 보정해주거나, 학습때 Noise를 더 늘려 다양한 데이터를 수집하도록 함으로써 어느정도 해결할 수 있다. 하지만 Self-Play가 가져올 수 있는 데이터 편향 문제는 여전히 존재하기 때문에 고민해볼 필요가 있다. 예를 들어 학습이 끝난 이후에 직접 소규모 Dataset에 대해서 지도학습을 한다거나, 혹은 Reward를 수정해서 적 진영에 침투하는 것과 방어하는 것을 목적으로 미세 조정을 위한 Self-Play를 진행하는 등 다양한 방법에 대해서 생각해보면 좋을 것 같다.



### 후기
딥러닝을 처음 공부하고 Resnet 공부까지 마친 상황에서 처음으로 진행해본 프로젝트였다. 3주 동안 밤을 새면서 공부하고 구현했는데 그 과정이 너무 재밌었고 결과를 확인하는 과정에서 엄청난 성취감이 느껴지기도 했었다. 또 개인적으로 십이장기의 전략이나 연구된 수가 궁금했는데 직접 구현한 인공지능으로 확인할 수 있어서 궁금증도 해소될 수 있었던 프로젝트였다. 강화학습의 재미를 느낄 수 있었던 좋은 기회였다.