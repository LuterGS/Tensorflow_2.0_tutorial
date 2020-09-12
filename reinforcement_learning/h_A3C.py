"""
A3C (Asynchronous advantage actor-critic)
DQN의 상위호환

Asynchronous
    여러 에이전트들이 비동기적으로 각자 자신의 환경에서 상호작용함

Actor-critic
    가치함수 V(s) - 어떤 상태가 얼마나 좋은지
    정책 ㅠ(s) - 일련의 액션의 확률 출력값
    2개를 모두 평가함
    -> 네트워크의 최상단부에 별도의 완전연결 계층으로 존재

    에이전트는 추정값을 비판적으로 이용하여, 정책을 더 지능적으로 업데이트함

Advantage
    에이전트의 행동이 좋은지 평가하는, 경험에 의거한 할인된 보상을 제공해줬었는데 (QN 관련)
    할인된 보상 대신 어드밴티지 추정값 사용
    -> 액션이 좋은지만 결정하는게 아니라, 얼마나 더 좋은 결과를 냈는지도 알아야해서
    -> 네트워크의 예측이 부족한 부분에 집중 가능
"""