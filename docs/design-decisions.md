# 설계 결정 기록

## 1. 무제한 queue 대신 bounded queue

**문제:** 입력률이 처리량보다 높아지면 queue가 계속 증가해 메모리와 tail latency가 통제되지 않습니다.

**선택:** queue capacity를 고정하고 초과 요청은 429로 빠르게 거절합니다.

**트레이드오프:** 순간 성공률을 일부 포기하지만 서버 전체가 느려지는 대신 admission boundary를 유지합니다.

**검증:** 강제 overload 시 60건 중 일부만 admission되고 나머지는 429가 되는 시나리오를 release gate에 포함했습니다.

## 2. scale-out 전에 dynamic batching

**문제:** inference call당 고정 비용이 있으면 요청마다 모델을 호출할 때 처리량이 낮습니다.

**선택:** 짧은 batch window를 두고 여러 요청을 하나의 backend call로 합칩니다.

**트레이드오프:** batch를 모으기 위한 수 ms의 대기 시간이 추가됩니다. 따라서 max wait와 client deadline 중 더 이른 시점을 batch deadline으로 사용합니다.

**검증:** 동일 deterministic workload에서 batch size 1과 8을 비교하고 throughput/p95/success rate를 기록합니다.

## 3. 무조건 retry 대신 circuit breaker + fallback

**문제:** 느리거나 장애 난 모델에 retry를 계속 보내면 retry storm과 장애 전파가 발생할 수 있습니다.

**선택:** primary timeout과 failure threshold를 넘으면 circuit을 열고 fallback으로 전환합니다.

**트레이드오프:** fallback 품질이 primary보다 낮을 수 있습니다. 대신 응답 가능성과 장애 격리를 우선합니다.

**검증:** primary 3회 실패 -> open -> fallback -> recovery probe -> closed를 자동화했습니다.

## 4. request timeout만 두지 않고 queue deadline 보존

**문제:** HTTP client가 이미 timeout됐는데 queue item이 살아 있으면 응답할 수 없는 요청이 backend 자원을 소비합니다.

**선택:** enqueue 시 deadline을 저장하고 dequeue/model call 직전에 만료 여부를 확인합니다.

**트레이드오프:** queue 로직이 복잡해지지만 불필요한 inference를 줄일 수 있습니다.

## 5. request span의 parent-child 대신 batch span link

**문제:** 여러 HTTP 요청이 하나의 batch로 합쳐지므로 하나의 parent를 선택하면 관계를 왜곡합니다.

**선택:** 각 request span context를 batch span에 OpenTelemetry Link로 연결합니다.

**트레이드오프:** trace UI 해석이 일반적인 단일 트리보다 조금 복잡하지만 fan-in 관계를 정확하게 표현합니다.

## 6. 한 프로세스 여러 worker보다 pod scale-out

한 프로세스 안에서 Uvicorn worker를 여러 개 띄우면 worker마다 queue와 batcher가 분리됩니다. 이 프로젝트는 한 pod에 worker 1개를 두고 Kubernetes replica로 scale-out하는 방식을 기본으로 합니다. 이렇게 하면 queue saturation metric과 batching behavior를 pod 단위로 해석하기 쉽습니다.
