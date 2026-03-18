# ACER (Actor-Critic with Experience Replay) - CartPole 구현

## 📌 프로젝트 개요

본 프로젝트는 강화학습 알고리즘인 ACER(Actor-Critic with Experience Replay)를 조사하고, 이를 기반으로 CartPole 환경에서 직접 재구현하는 것을 목표로 합니다.  
ACER는 기존 Actor-Critic 계열 알고리즘의 구조를 유지하면서 Experience Replay를 결합하고, Off-policy 학습을 안정적으로 수행하기 위한 기법을 추가한 알고리즘입니다.

기존 Actor-Critic, A2C, A3C 알고리즘은 모두 on-policy 방식으로 동작하기 때문에, 한 번 사용한 데이터를 다시 활용할 수 없다는 한계를 가지고 있습니다. 이는 학습 효율성을 떨어뜨리는 주요 원인입니다.  
ACER는 Replay Buffer를 도입하여 과거 데이터를 재사용할 수 있도록 하고, Importance Sampling을 통해 정책 간 분포 차이를 보정하여 안정적인 학습을 가능하게 합니다.

본 프로젝트에서는 이러한 ACER의 핵심 개념을 CartPole 환경에서 구현하고, 실제로 동작하는 코드를 통해 알고리즘의 동작 원리를 이해하는 것을 목표로 합니다.

---

## 📌 ACER 알고리즘의 등장 배경

기존 Actor-Critic 계열 알고리즘은 정책 기반 방법과 가치 기반 방법의 장점을 결합한 구조로, 안정적인 학습을 가능하게 합니다. 그러나 on-policy 방식으로 인해 과거 데이터를 재사용할 수 없으며, 매번 새로운 데이터를 환경으로부터 수집해야 합니다.  
이러한 구조는 샘플 효율성이 낮다는 문제를 가지며, 실제 환경에서는 데이터 수집 비용이 큰 부담으로 작용합니다.

이 문제를 해결하기 위해 Experience Replay를 도입하려는 시도가 이루어졌습니다. Experience Replay는 과거 데이터를 저장하고 이를 반복적으로 학습에 사용하는 방식으로, 데이터 효율성을 크게 향상시킬 수 있습니다.  
하지만 Actor-Critic 계열 알고리즘에 이를 적용할 경우, 과거 정책과 현재 정책 간의 분포 차이가 발생하는 off-policy 문제가 발생합니다.

ACER는 이러한 문제를 해결하기 위해 등장한 알고리즘으로, Experience Replay와 Importance Sampling을 결합하여 데이터 효율성과 학습 안정성을 동시에 확보합니다.

---

## 📌 Actor-Critic / A2C / A3C와의 차이

Actor-Critic은 가장 기본적인 형태로, 단일 환경에서 데이터를 수집하며 학습을 진행합니다.  
A2C는 여러 환경을 동시에 실행하여 데이터를 병렬로 수집하고, 이를 동기적으로 업데이트합니다.  
A3C는 비동기 방식으로 여러 프로세스가 독립적으로 학습을 수행하며, 글로벌 모델을 업데이트합니다.

이 세 알고리즘은 모두 on-policy 기반으로 동작하기 때문에, 과거 데이터를 재사용할 수 없다는 공통적인 한계를 가지고 있습니다.

반면 ACER는 Replay Buffer를 도입하여 과거 데이터를 재사용할 수 있으며, off-policy 학습이 가능하다는 점에서 큰 차이를 보입니다.  
또한 Importance Sampling과 Truncated 기법을 통해 정책 간 분포 차이를 보정하여 안정적인 학습을 수행합니다.

---

## 📌 핵심 개념

### 1. Experience Replay

Experience Replay는 환경에서 수집한 데이터를 Replay Buffer에 저장하고, 이를 무작위로 샘플링하여 학습에 사용하는 기법입니다.  
이를 통해 데이터 재사용이 가능해지고, 학습 효율성이 크게 향상됩니다.

---

### 2. Importance Sampling

Replay Buffer에 저장된 데이터는 과거 정책에 의해 생성된 것이므로, 현재 정책과 분포 차이가 발생합니다.  
이를 보정하기 위해 Importance Sampling을 사용합니다.

ρ = π(a|s) / μ(a|s)


여기서 π는 현재 정책, μ는 데이터를 생성한 과거 정책을 의미합니다.  
이 값을 통해 과거 데이터를 현재 정책 기준으로 재가중할 수 있습니다.

---

### 3. Truncated Importance Sampling

Importance Sampling 값이 지나치게 커질 경우 학습이 불안정해질 수 있습니다.  
이를 방지하기 위해 clipping을 적용합니다.

ρ̄ = min(c, ρ)


이 기법은 학습의 분산을 줄이고 안정성을 확보하는 데 중요한 역할을 합니다.

---

## 📌 구현 내용 (CartPole ACER)

본 프로젝트에서는 CartPole 환경을 기반으로 ACER 알고리즘을 재구현하였습니다.

### ✔ 환경
- CartPole-v1
- 상태: 4차원 벡터
- 행동: 2개 (왼쪽, 오른쪽)

---

### ✔ 네트워크 구조

Actor-Critic 구조를 사용하여 정책과 가치를 동시에 학습합니다.

- 입력: 상태 (4차원)
- 은닉층: 256 노드
- 출력:
  - Actor: 행동 확률 (2)
  - Critic: 상태 가치 (1)

---

### ✔ Replay Buffer

Replay Buffer를 사용하여 상태, 행동, 보상, 다음 상태, 행동 확률을 저장합니다.

(state, action, reward, next_state, done, behavior_prob)


behavior_prob는 해당 행동을 선택했을 당시의 정책 확률이며, Importance Sampling 계산에 사용됩니다.

---

### ✔ 학습 과정

1. 환경에서 데이터를 수집합니다  
2. Replay Buffer에 데이터를 저장합니다  
3. Replay Buffer에서 데이터를 샘플링합니다  
4. Importance Sampling을 적용합니다  
5. Actor와 Critic을 업데이트합니다  

---

### ✔ Loss 함수

Policy Loss:  -ρ̄ * log(π(a|s)) * advantage

Value Loss:  smooth_l1_loss(V(s), target)


---

## 📌 실행 방법

```bash
pip install gymnasium torch numpy
python acer_cartpole.py

---

## 📌 로컬 실행 방법

```bash
pip install gymnasium torch numpy
python3 acer_cartpole.py

📌 결과

CartPole 환경에서 안정적으로 학습이 진행됩니다.

Replay Buffer를 통해 데이터 효율성이 향상됩니다.

Importance Sampling을 통해 off-policy 문제를 보정합니다.

📌 결론

ACER는 Actor-Critic 구조를 유지하면서 Experience Replay를 도입하여 데이터 효율성을 크게 향상시킨 알고리즘입니다.
또한 Importance Sampling과 Truncated 기법을 통해 off-policy 문제를 안정적으로 해결합니다.

본 프로젝트를 통해 강화학습에서 데이터 재사용의 중요성과, 정책 간 분포 차이를 보정하는 과정이 학습 안정성에 미치는 영향을 이해할 수 있었습니다.