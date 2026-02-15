---
title: "[논문 리뷰] Deep Residual Learning for Image Recognition"
tags: "인공지능"
---

**원 논문:** <https://arxiv.org/abs/1512.03385>
**직접 구현:** <https://github.com/studixxne/ResNet-impl>

### Introduction
Network의 표현력은 Stacked Layer의 깊이가 깊어질수록 풍부해진다. 
Feature들을 레이어들이 더 많이 분리할 수 있기 때문이다.
실제로 이전까지 ImageNet Challenge에서 VGG같은 깊은 신경망이 성능이 더 좋다는 것을 보여주고 있었다.

'그렇다면 Depth의 중요성에 따라 더 많은 Layer들을 쌓으면 쌓을수록 더 나은 네트워크가 될 수 있을까?'

이 질문으로부터 이 논문이 시작하게 된다.

하지만 이러한 질문에 대해서 문제점이 존재한다.

**(1) Vanishing/Exploding Gradients Problem**
층이 깊어지면서 활성화 함수로 인해 역전파 과정에서 기울기가 점점 사라지거나,
기울기가 기하급수적으로 커져 학습이 불가능하게 되어버리는 문제가 발생한다.
이러한 문제들은 학습 초기부터 학습하는 것을 방해한다.

이러한 문제를 해결하기 위해서 Normalized Initialiation(He Init)과 Intermediate Normalization Layer(BatchNorm Layer)를 사용하게 되었고 이를 통해서 문제를 해결할 수 있게 되었다.

그렇다면 기울기 소실/폭발 문제를 해결했으니 층을 깊게 쌓으면 표현력이 증가하여 문제를 해결할 수 있을까?
하지만 실제로 층을 깊게 쌓아서 성능을 측정해본 결과 오히려 성능이 나빠지게 된다!

표현력이 너무 풍부해져서 Overfitting의 문제가 아닐까?
-> 아니다! Test Data에 대한 정확도 뿐만 아니라 Training Data에 대해서도 정확도가 낮아졌기 때문이다!

이를 'Degradation Problem'이라고 한다.

**(2) Degradation Problem**
층을 깊게 쌓으면 쌓을 수록 오히려 Test, Training Data 모두 정확도가 감소하는 문제이다.
이러한 문제는 Overfitting으로 발생하는 것이 아니며, 기울기 소실/폭발 문제로도 발생하는 것이 아니다.

실제로 이미 학습된 얇은 층에 Identity Layer를 추가해서 더 깊은 네트워크를 만들었을 때, 우리의 예측은 표현력이 증가하였음으로 Training Data에 대해서 최소한 얇은 층이 가진 정확도를 유지하거나 정확도가 더 올라야 한다. 그러나 실제로 실험 결과를 확인했을 때에는 오히려 떨어진 것을 확인할 수 있다.

이는 최적화가 잘 이루어지지 않았다는 것을 의미하며 결과적으로 모델의 깊이가 증가할수록 시스템의 최적화가 어려워진다는 것을 시사한다.

그래서 이러한 문제를 해결하기 위해 논문에서는 **'Residual Learning'** 을 사용하게 된다.

### Deep Residual Learning

**Residual Learning**은 원하는 Mapping을 그대로 쌓아 올려 학습하는 방식 대신, 입력 값과 최종 목표값의 차이인 Residual Mapping을 학습시키도록 하는 방식이다.

예를 들어 $H(x)$가 우리가 최종적으로 얻고 싶어하는 결과라면, $H(x)$를 바로 뽑아내도록 학습 시키는 것이 아니라 $F(x)=H(x)-x$로 학습시키는 것이다. 그렇게 되면 결과적으로 $H(x) = F(x) + x$로 우리가 원하는 결과를 얻을 수 있는 것이다.

이러한 방식으로 인해 $x$와 $H(x)$와의 **차이**만 학습시키면 되기 때문에 신경망을 더 쉽게 최적화할 수 있다.

극단적인 예시로 위에서 들었던 것처럼 이미 $H(x)$를 학습한 얇은 모델에서 Identity를 여러 개 쌓아놓았을 때 기존의 방법이라면 여러 개의 가중치가 비선형성을 거치며 $H(x)$를 유지하도록 학습해야 하지만, Residual 방식이라면 단순히 $F(x)$가 0이 되도록 학습하면 된다.

$H(x)$를 유지하기 위해 여러 개의 파라미터를 최적화시키는 것 보단 단순히 $F(x)$를 0으로 최적화시키는 것이 학습에 훨씬 유리할 것이다. 이는 Residual 방식이 더 효율적인 것을 보여주는 예시이다.

돌아와서, 결론적으로 얻고자 하는 $H(x)$는 $F(x) + x$로 얻을  수 있다.
이 때 $x$를 더하는 과정은 어떻게 구현할 수 있을까?

이는 **"Shortcut Connections"** 으로 구현할 수 있다.

**Shortcut Connection**은 Identity Mapping으로 한 개 이상의 레이어들을 스킵하여 $x$를 $F(x)$와 더해주는 역할을 한다.

이를 식으로 표현하면 $y = F(x, {W_i}) + x$로 표현하며

만일 $x$와 $F(x, {W_i})$의 차원이 다르다면 $y = F(x, {W_i}) + {W_s}x$로 표현할 수 있다.
여기서 ${W_s}$는 차원을 맞추기 위한 1\*1 Conv를 통한 Projection 연산이다.

![Basic Block](/post_image/2026/0216/figure1.png)


결과적으로 이를 적용함으로써 층이 매우 깊더라도 차이만 학습하면 되기 때문에 최적화가 훨씬 쉬워지게 되며 논문의 실험 결과도 이러한 결과를 증명한다.

여기서 Shortcut Connection을 전체적인 네트워크 구조에서 좀 더 직관적으로 본다면, 여러 개의 네트워크가 모여서 하나의 새로운 네트워크가 될 때, 작은 Network들이 하나의 Network로 Degradation 문제 없이 잘 합쳐지도록 하는 역할을 한다고 이해를 하면 더 좋을 것 같다.

### Network Architectures
![Networks](/post_image/2026/0216/figure2.png)

논문에서는 이전에 나온 VGG의 영감을 통해 PlainNet을 설계하였다.
여기서 ResNet은 PlainNet에 Shortcut Connections을 추가한 네트워크이다.
PlainNet과 ResNet의 성능 비교를 통해 Residual Learning이 얼마나 성능이 좋은지 확인해보자.

![Compare](/post_image/2026/0216/figure3.png)

![Table](/post_image/2026/0216/figure4.png)

ImageNet에 대한 실험 결과

PlainNet의 경우 층이 깊어짐에 따라 학습 속도가 느리고 Error가 더 높은 것을 확인할 수 있다.
이러한 점에서 Degradation Problem이 드러난다.

반면 ResNet의 경우 PlainNet과 반대의 결과를 보인다.
오히려 층이 깊을 수록 학습 속도가 빠르고 Error가 더 낮은 것을 보여준다!
Residual Learning이 Degradation Problem을 해결하고 층이 깊어질 수록 더 좋은 성능을 내는 것을 증명한 것이다!

또한 동일한 깊이더라도 ResNet이 PlainNet보다 더 빠른 학습 속도를 보여주는 것을 확인할 수 있다.

### Deeper BottleNeck Architectures
층이 너무 깊어진 경우에는 연산량이 매우 많아져서 Training Time이 매우 길어질 수 있다.
이를 보완하기 위해 논문에서는 BottleNeck Architecture를 제안하였다.

![BottleNeck](/post_image/2026/0216/figure5.png)

BottleNeck Block의 경우 층이 깊어질 경우 연산량을 줄이기 위해 1\*1 Conv를 통해 차원을 줄이고 연산 후에 다시 증폭하는 과정을 포함했다. 그리고 3\*3 Conv 연산을 한 번만 수행한다.

결과적으로 3\*3 Conv의 개수가 줄어들어 학습해야 하는 파라미터가 줄어드므로 Basic Block보다 더 적은 연산으로 학습을 진행할 수 있다.

### Implementation For CIFAR-10 

논문에서는 주로 ImageNet Dataset에 대해서 수행하였지만, 로컬로 가볍게 실험 결과를 확인해보기 위해서 CIFAR-10에 대해 적힌 부문에 대해서 직접 Block들을 구현해보고 ResNet을 구현해보았다.

Shortcut의 경우 논문 본문에서는 파라미터가 없는 Zero-Padding 방식으로 구현했으나, 
실 구현에서는 Conv 1*1을 통해 Projection으로 차원을 맞춰주는 방식으로 구현하였다.

CIFAR-10 Dataset을 위한 Block들의 구현체는 다음과 같다.

``` python
class PlainBlockForCIFAR(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()

        self.h_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.h_layer(x)
        return h
```

PlainBlock의 경우에는 단순히 Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU을 하나의 묶음으로 구현해주었다.

<br>

``` python
class BasicBlockForCIFAR(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()

        self.f_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        if in_ch != out_ch:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)            
            )
            
        else:
            self.short_cut = nn.Identity()

    def forward(self, x):
        identity = self.short_cut(x)
        f = self.f_layer(x)
        h = f + identity
        h = F.relu(h)
        return h
```

반면 Shortcut Connection이 적용된 BasicBlock은 short_cut layer를 만들어주었다.

이 때 in_channel과 out_channel의 크기가 다르다면 feature map size가 절반으로 줄어들고 out_channel의 크기가 in_channel의 크기보다 2배 증가했음으로 Conv를 통해 차원을 일치시켜주도록 했다.

결과적으로 f_layer를 통과시켜 f를 구한 뒤, short_cut을 통과시켜 얻은 x를 더해줌으로써 $F(x) + x$를 계산하였고 **덧셈 연산 이후에 ReLU**를 적용시켜주었다.


<br>

ResNet For CIFAR 모델은 다음과 같다.
``` python
class ResNetForCIFAR(nn.Module):
    def __init__(self, block, block_nums):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv2 = self._make_layer(block, block_nums[0], 16, 16, 1)
        self.conv3 = self._make_layer(block, block_nums[1], 16, 32, 2)
        self.conv4 = self._make_layer(block, block_nums[2], 32, 64, 2)

        self.last_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

        self._init_He_weights()

    def _make_layer(self, block, block_num, in_ch, mid_ch, stride):
        layers = []
        layers.append(block(in_ch, mid_ch, stride=stride))

        for _ in range(block_num-1):
            layers.append(block(mid_ch, mid_ch, stride=1))
        
        return nn.Sequential(*layers)
    
    def _init_He_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.last_layer(x)
        return x
```
Block의 종류와 Blcok 개수를 입력받아 ResNet을 생성할 수 있다.

Conv1 -> Conv2 -> Conv3 -> Conv4 -> Last_layer 순서로 이루어져 있으며
Input Shape (3, 32, 32)에서 (16, 32, 32) -> (32, 16, 16) -> (64, 8, 8) -> (64, 1)로 만들어 fc layer를 통과하도록 구현했다.

가중치 초기화는 He Initialization으로 하였다.



### 직접 구현해본 실험 결과
##### Test Error(%)
![PlainNet_Compare](/post_image/2026/0216/figure6.png)
![ResNet_Compare](/post_image/2026/0216/figure7.png)

PlainNet-20, PlainNet-56의 비교 그래프와 ResNet-20, ResNet-56의 비교 그래프를 확인해보았을 때 CIFAR-10에 대한 데이터셋에도 깊이에 따른 역전 효과가 잘 드러나는 것을 확인할 수 있었다.

실제로 PlainNet의 경우 깊이가 깊어질수록 얇은 모델보다 크게 진동하며 수렴 속도가 늦어졌으며 Error도 확연하게 증가한 것을 확인할 수 있다.

그 반면 Shortcut Connection을 적용한 ResNet의 경우에는 깊이가 깊어져도 학습을 안정적으로 잘하는 것을 확인할 수 있으며 오히려 깊이가 얇은 모델보다 더 좋은 정확도를 보이는 것을 확인할 수 있었다.

이를 통해 논문의 내용과 동일하게 실제로 Shortcut Connection을 통해서 Degradation 문제를 다룰 수 있으며 더 쉽게 모델을 최적화할 수 있다는 것을 확인할 수 있었다.

<br>

![Total_PlainNet_Compare](/post_image/2026/0216/figure8.png)
![Total_ResNet_Compare](/post_image/2026/0216/figure9.png)

모든 N = 3, 5, 7, 9에 대해서 실험하였을 때도 마찬가지로 PlainNet의 경우에는 깊이가 깊어질수록 Degradation 문제가 발생하는 경향을 분명하게 드러내고 있었으며 층이 깊어질수록 그래프가 진동하는 것을 보았을 때 최적화가 점점 더 어려워진다는 것을 확인할 수 있다.

그 반면 ResNet의 경우에는 깊이가 깊어지더라도 동일하게 안정적으로 학습하는 것을 확인할 수 있으며 실제로 깊이가 깊어질수록 안정적으로 학습하며 좋은 성능을 내는 것을 확인할 수 있다.

<br>

#### 모든 네트워크의 비교 그래프
![Total_Compare](/post_image/2026/0216/figure10.png)


최종적으로 **Residual Learning**과 **Shortcut Connection**은 모델이 **Residual**만을 학습하도록 한다. 이는 모델의 최적화 난이도를 확연하게 낮춤으로써 **깊이의 제약을 극복**할 수 있다는 것을 보여준다.