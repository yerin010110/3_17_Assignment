# ============================================================
# ACER (Actor-Critic with Experience Replay) for CartPole-v1
# ------------------------------------------------------------
# 이 코드는 과제 제출용으로 작성한 CartPole 환경용 ACER 형태 구현입니다.
#
# [중요]
# - 기존 A3C 스타일(멀티프로세스 비동기 학습) 코드를
#   ACER 스타일(Replay Buffer + Off-policy 보정)로 변환한 버전입니다.
# - 실행 가능하도록 단일 프로세스 기반으로 재구성했습니다.
# - ACER의 핵심 개념인 Experience Replay, Importance Sampling,
#   Truncated Importance Sampling을 반영했습니다.
#
# [논문 원형 ACER와의 차이]
# - 본 코드는 교육/과제용 구현입니다.
# - 논문 원형 ACER의 Retrace, bias correction, trust region update까지
#   모두 완전 구현한 것은 아닙니다.
# - 대신 과제 핵심 요구인 "왜 ACER가 필요한지"를 코드로 보여줄 수 있도록
#   Replay Buffer + Off-policy correction 중심으로 설계했습니다.
# ============================================================

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 1. 하이퍼파라미터 설정
# ============================================================
learning_rate = 0.0005        # 학습률
gamma = 0.98                  # 할인율
buffer_limit = 50000          # Replay Buffer 최대 크기
batch_size = 32               # 한 번 업데이트할 때 사용할 샘플 수
max_train_episodes = 400      # 최대 학습 에피소드 수
max_test_episodes = 10        # 테스트 에피소드 수
print_interval = 10           # 로그 출력 간격
replay_ratio = 4              # 환경에서 한 번 step한 뒤 replay update를 몇 번 할지
warmup_steps = 1000           # Replay Buffer가 이만큼 쌓이기 전까지는 본격 학습 X
rho_clip = 10.0               # Truncated Importance Sampling clipping 값
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Replay Buffer 정의
# ============================================================
class ReplayBuffer:
    """
    ACER의 핵심 요소 중 하나인 Experience Replay를 위한 버퍼입니다.

    저장하는 transition 형식:
    (state, action, reward, next_state, done_mask, behavior_prob)

    여기서 behavior_prob는
    "그 행동을 실제로 선택했을 당시의 정책 확률(mu(a|s))"입니다.

    ACER는 과거 정책으로 수집된 데이터를 다시 학습에 사용하므로,
    현재 정책 pi(a|s)와 과거 정책 mu(a|s)의 차이를 Importance Sampling으로
    보정해야 합니다. 따라서 behavior_prob 저장이 매우 중요합니다.
    """

    def __init__(self, capacity=buffer_limit):
        self.buffer = deque(maxlen=capacity)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, mu_prob_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask, mu_prob = transition

            s_lst.append(s)
            a_lst.append([a])               # gather 연산을 위해 [a] 형태로 저장
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            mu_prob_lst.append([mu_prob])   # behavior policy 확률 저장

        return (
            torch.tensor(s_lst, dtype=torch.float, device=device),
            torch.tensor(a_lst, dtype=torch.long, device=device),
            torch.tensor(r_lst, dtype=torch.float, device=device),
            torch.tensor(s_prime_lst, dtype=torch.float, device=device),
            torch.tensor(done_mask_lst, dtype=torch.float, device=device),
            torch.tensor(mu_prob_lst, dtype=torch.float, device=device),
        )

    def size(self):
        return len(self.buffer)


# ============================================================
# 3. Actor-Critic 네트워크 정의
# ============================================================
class ActorCritic(nn.Module):
    """
    CartPole 상태는 4차원입니다.
    행동은 왼쪽(0), 오른쪽(1) 두 가지입니다.

    - pi(): Actor 역할
      상태를 입력받아 각 행동의 확률을 출력합니다.
    - v(): Critic 역할
      상태를 입력받아 해당 상태의 가치 V(s)를 출력합니다.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


# ============================================================
# 4. ACER 스타일 업데이트 함수
# ============================================================
def train_acer(model, memory, optimizer):
    """
    Replay Buffer에서 과거 데이터를 샘플링하여
    ACER 스타일로 Actor-Critic을 업데이트합니다.

    핵심 흐름:
    1) 현재 정책 pi(a|s) 계산
    2) 과거 정책 mu(a|s)와 비교하여 importance ratio 계산
       rho = pi(a|s) / mu(a|s)
    3) rho를 clipping하여 truncated importance sampling 적용
    4) Critic은 TD target으로 학습
    5) Actor는 clipped importance weight를 반영한 policy loss로 학습

    참고:
    - 논문 원형 ACER는 Retrace, bias correction 등을 추가로 사용합니다.
    - 본 코드는 과제용 구현으로 핵심 아이디어 위주로 재현합니다.
    """

    if memory.size() < batch_size:
        return None, None, None

    s, a, r, s_prime, done_mask, mu_prob = memory.sample(batch_size)

    # --------------------------------------------------------
    # (1) 현재 정책 pi(a|s) 계산
    # --------------------------------------------------------
    # s의 shape: (batch_size, 4)
    # pi의 shape: (batch_size, 2)
    pi = model.pi(s, softmax_dim=1)

    # 실제 선택된 행동 a에 대한 현재 정책 확률 pi(a|s)만 추출
    pi_a = pi.gather(1, a)

    # 수치 안정성 확보
    pi_a = torch.clamp(pi_a, min=1e-8)
    mu_prob = torch.clamp(mu_prob, min=1e-8)

    # --------------------------------------------------------
    # (2) Importance Sampling Ratio 계산
    # --------------------------------------------------------
    # rho = pi(a|s) / mu(a|s)
    # 현재 정책과 과거 정책의 차이를 보정하는 비율
    rho = pi_a / mu_prob

    # --------------------------------------------------------
    # (3) Truncated Importance Sampling
    # --------------------------------------------------------
    # rho 값이 지나치게 커지면 분산이 폭발할 수 있으므로 clip 적용
    rho_bar = torch.clamp(rho, max=rho_clip)

    # --------------------------------------------------------
    # (4) Critic 업데이트용 TD Target 계산
    # --------------------------------------------------------
    # V(s'), V(s) 계산
    v_s = model.v(s)
    v_s_prime = model.v(s_prime).detach()

    # done_mask:
    # - 에피소드가 끝났으면 0.0
    # - 아직 안 끝났으면 1.0
    td_target = r + gamma * v_s_prime * done_mask

    # Advantage = TD target - V(s)
    advantage = td_target - v_s

    # --------------------------------------------------------
    # (5) Policy Loss 계산
    # --------------------------------------------------------
    # 기본 Actor-Critic의 정책 손실:
    #   -log(pi(a|s)) * advantage
    #
    # ACER 스타일 보정:
    #   -rho_bar * log(pi(a|s)) * advantage
    #
    # rho_bar를 곱하는 이유:
    # 과거 정책으로 생성된 데이터를 현재 정책 기준으로 보정하기 위함
    policy_loss = -(rho_bar.detach() * torch.log(pi_a) * advantage.detach())

    # --------------------------------------------------------
    # (6) Value Loss 계산
    # --------------------------------------------------------
    # Critic은 TD target에 맞게 상태 가치를 학습
    value_loss = F.smooth_l1_loss(v_s, td_target)

    # --------------------------------------------------------
    # (7) 전체 손실 결합 후 역전파
    # --------------------------------------------------------
    loss = policy_loss.mean() + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.mean().item(), value_loss.item()


# ============================================================
# 5. 테스트 함수
# ============================================================
def test(model, env_name="CartPole-v1", episodes=max_test_episodes):
    """
    학습된 모델을 별도 환경에서 테스트하는 함수입니다.
    """
    test_env = gym.make(env_name)
    model.eval()

    total_score = 0.0

    with torch.no_grad():
        for _ in range(episodes):
            s, _ = test_env.reset()
            done = False
            episode_score = 0.0

            while not done:
                state = torch.from_numpy(s).float().to(device)

                # 테스트 시에는 확률이 가장 큰 행동을 선택하여
                # 정책이 얼마나 안정적으로 학습되었는지 확인
                prob = model.pi(state)
                action = torch.argmax(prob).item()

                s_prime, r, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated

                s = s_prime
                episode_score += r

            total_score += episode_score

    test_env.close()
    model.train()

    return total_score / episodes


# ============================================================
# 6. 메인 학습 루프
# ============================================================
def main():
    env = gym.make("CartPole-v1")
    memory = ReplayBuffer()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    score = 0.0
    total_step = 0

    print("=" * 60)
    print("ACER 스타일 CartPole 학습 시작")
    print(f"device: {device}")
    print("=" * 60)

    for n_epi in range(1, max_train_episodes + 1):
        s, _ = env.reset()
        done = False
        episode_score = 0.0

        while not done:
            state = torch.from_numpy(s).float().to(device)

            # ------------------------------------------------
            # (1) 현재 정책으로 행동 확률 계산
            # ------------------------------------------------
            prob = model.pi(state)

            # numpy 변환 전 detach 필요
            prob_np = prob.detach().cpu().numpy()

            # ------------------------------------------------
            # (2) 행동 샘플링
            # ------------------------------------------------
            # CartPole은 행동이 2개이므로 확률 분포에 따라 샘플링
            action = np.random.choice(2, p=prob_np)

            # behavior policy 확률 저장
            # 이 값이 이후 importance sampling의 분모 mu(a|s)가 됨
            behavior_prob = prob_np[action]

            # ------------------------------------------------
            # (3) 환경 진행
            # ------------------------------------------------
            s_prime, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # done_mask:
            # 종료되었으면 0.0, 계속 진행되면 1.0
            done_mask = 0.0 if done else 1.0

            # 보상 스케일링
            # CartPole은 step당 보상 1을 주므로 너무 큰 값으로 누적되지 않도록 조절 가능
            scaled_r = r / 100.0

            # ------------------------------------------------
            # (4) Replay Buffer에 transition 저장
            # ------------------------------------------------
            memory.put((s, action, scaled_r, s_prime, done_mask, behavior_prob))

            # ------------------------------------------------
            # (5) 상태 갱신
            # ------------------------------------------------
            s = s_prime
            episode_score += r
            score += r
            total_step += 1

            # ------------------------------------------------
            # (6) Replay Buffer가 충분히 쌓였으면
            #     replay update 수행
            # ------------------------------------------------
            if memory.size() > warmup_steps:
                for _ in range(replay_ratio):
                    train_acer(model, memory, optimizer)

        # ----------------------------------------------------
        # 에피소드 단위 로그 출력
        # ----------------------------------------------------
        if n_epi % print_interval == 0:
            avg_train_score = score / print_interval
            avg_test_score = test(model)

            print(
                f"[Episode {n_epi:4d}] "
                f"평균 Train Score: {avg_train_score:7.2f} | "
                f"평균 Test Score: {avg_test_score:7.2f} | "
                f"Buffer Size: {memory.size():5d}"
            )
            score = 0.0

    env.close()

    # --------------------------------------------------------
    # 최종 테스트
    # --------------------------------------------------------
    final_test_score = test(model, episodes=20)
    print("=" * 60)
    print(f"최종 평균 테스트 점수: {final_test_score:.2f}")
    print("=" * 60)

    # --------------------------------------------------------
    # 모델 저장
    # --------------------------------------------------------
    torch.save(model.state_dict(), "acer_cartpole.pth")
    print("학습된 모델을 'acer_cartpole.pth'로 저장했습니다.")


# ============================================================
# 7. 실행 진입점
# ============================================================
if __name__ == "__main__":
    main()
