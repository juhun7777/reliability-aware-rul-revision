# BiLSTM 보완 실험과 Transformer 추가 검토

2026년 10월 26일 갱신

기존 논문의 PHI 기반 Bi-LSTM 전문가와 조건 유사도에 따른 가중 결합
아이디어를 중심으로 보완했습니다. 같은 개선 방법을 Transformer에도
적용했을 때 낮은 예측 오차가 관찰되어, 추가 개선 방향으로 함께 정리했습니다.
Transformer로 주모델을 바꾸기로 확정한 것은 아닙니다.

## 기존 전문가를 유지한 게이트 보완

관측 신호가 학습한 0A·0.5A·1.0A 중 어느 조건과 유사한지 추정해 세
전문가의 출력을 결합합니다. PHI를 쓰는 전문가를 고정하고, 관측량으로
보정한 STFT DC 진폭·변화율 등으로 결합 가중치를 계산하는 게이트를
적용했습니다.

| 기존 전문가 구성과 게이트 | MAE 사이클 | RMSE 사이클 |
|:---|---:|---:|
| Bi-LSTM + PHI + 기존 STFT soft | 16,436 | 21,960 |
| 같은 Bi-LSTM 전문가 + 개선 게이트 | 9,301 | 11,308 |
| Transformer + PHI + 같은 개선 게이트 | 8,223 | 10,210 |

각각 seed 42·43·44의 평균입니다. 평가 대상은 같은 15파일의 70·80·90%
관측점 45개이며, 독립된 시편 45개를 뜻하지 않습니다. Bi-LSTM에서는
회귀 전문가를 그대로 두고 게이트를 바꿨을 때 MAE가 약 43.4% 감소했습니다.
PHI는 세 행 모두 전문가 입력에 포함됩니다. 게이트 자체는 관측 저항과
사이클을 사용하고 PHI를 직접 입력받지는 않습니다.

개선 게이트의 중심인 DC 성분은 Hann 가중 이동평균과 같은 정보입니다.
따라서 여기서 확인한 것은 조건 유사도에 따른 결합의 개선 가능성입니다.
비DC 주파수 특징의 추가 효과는 별도 대조가 필요합니다. 이 단계의 고정
전문가는 과거 val로 선택된 모델이며, 두 backbone의 크기·학습 조건까지
완전히 통일한 비교는 아닙니다.

## 별도로 재학습한 Transformer 비교

후속 단계에서는 학습 내부 원본 그룹으로 epoch와 학습률 일정을 선택하고,
전체 학습자료로 다시 학습했습니다. 아래는 이 별도 절차에서 seed 3개의
평균 MAE·RMSE를 계산한 값입니다. 관측점별 MC dropout 30회의 평균 예측을
평가했으며, 평가자료는 epoch·게이트 설정 선택에 사용하지 않았습니다.

| Transformer 구성 | MAE 사이클 | RMSE 사이클 |
|:---|---:|---:|
| 전체 데이터 단일 모델 | 10,521 | 12,795 |
| 공유형 3전문가 + STFT soft | 9,838 | 12,213 |
| 공유형 soft + 단일 모델 50 대 50 | 9,430 | 11,314 |

공유형은 전체 학습자료로 공통 Transformer 특징 추출부를 학습하고 전류별
출력부 3개를 유지하는 구조입니다. MAE는 정답과 예측의 평균 절대 차이로
낮을수록 좋습니다. 첫 표의 8,223·9,301과 이 표의 9,430은 학습·선택 및
추론 절차가 달라 직접 순위를 매기지 않습니다. 추가 대조 결과는
[전체 집계 CSV](../examples/development_results.csv)에 함께 남겼습니다.

## 코드와 설정의 대응

- `train.py`와 `models/lstm.py`: 기존 Bi-LSTM 계열의 보완된 학습 경로입니다.
  기본 실행이 첫 표의 개선 게이트 실험을 자동 재현하는 것은 아닙니다.
- `utils/condition_gate.py`: PHI를 사용하는 Bi-LSTM·Transformer 전문가
  어느 쪽과도 별도로 사용할 수 있는 저항 기반 유사도 게이트입니다.
- `train_revision.py`: 전체 데이터 단일 Transformer, 조건별 독립 전문가,
  공유형 전문가를 학습하고 비교합니다. `--models all`은 Transformer
  구조 세 가지를 뜻하며 Bi-LSTM을 함께 학습하는 옵션은 아닙니다.
- [설정 기록](../examples/development_transformer_config.json): 두 번째 표를
  만든 Transformer 실행의 설정입니다. JSON을 직접 읽는 `--config` 옵션은
  없으며, 명시한 CLI 인수로 실행합니다.

```bash
python train_revision.py --manifest /path/manifest.csv --output results/new_run --models all --seeds 42 43 44 --epochs 50 --patience 12 --batch-size 128 --fractions 0.7 0.8 0.9 --alpha 2.34 --beta 0.133 --mc-samples 30 --gate-bins dc --allow-development-overlap
```

위 명령은 원본 그룹이 겹치는 개발 자료를 명시적으로 실행하는 예입니다.
그룹이 겹치지 않는 평가에서는 `--allow-development-overlap`을 사용하지
않습니다. 별도로 전달한 데이터 ZIP에 manifest가 없다면 실제 원본 그룹과
분할을 기록한 파일을 먼저 준비해야 합니다. 폴더 이름만으로 원본 그룹이나
실측 전류를 자동 인증하지 않습니다.

공개 저장소는 기존 공개 샘플 18개와 코드를 제공합니다. 전체 150 CSV와
표의 고정 전문가 체크포인트는 공개 파일에 포함하지 않았으므로, 샘플만
실행하여 같은 수치가 나오는 것으로 설명하지 않습니다.

## 저장된 전문가 출력의 재계산

`tools/recompute_frozen_results.py`는 기존 모델이 저장한 결정론적 전문가
출력과 가중치로 예측·오차를 다시 계산합니다. Bi-LSTM·Transformer에 공통으로
사용할 수 있으며, 게이트를 새로 학습하거나 원시 측정값을 변경하지 않습니다.

입력 CSV의 필수 열은 다음과 같습니다.

```text
model,gate,seed,record_id,observed_cycle,target,expert0,expert1,expert2,weight0,weight1,weight2
```

전문가와 가중치의 순서는 0A·0.5A·1.0A입니다. `model`에는 backbone을,
`gate`에는 `original_soft`·`improved_soft` 등 비교할 방식을 적습니다.
같은 model·seed의 게이트들은 같은 관측점·정답·전문가 출력을 가져야 하며,
파일 순서를 바꿔도 ID와 관측 사이클로 대응을 확인합니다. 가중치 합이
1이 아니거나, 비교 중 전문가 출력이 달라지거나, 관측점이 빠지면 오류로
중단합니다. 선택적인 `prediction` 열은 재계산 결과와 대조합니다.
float32 저장 과정의 작은 차이는 상대 허용오차 1e-7·절대 허용오차
0.000001사이클 이내에서 허용하고, 저장된 예측값과 차이를 출력에 남깁니다.

```bash
python tools/recompute_frozen_results.py --predictions /path/frozen_components.csv --output results/frozen_audit
```

출력은 관측점별 예측, seed별 MAE·RMSE, seed 평균과 표본 표준편차,
입력 해시 및 계산 정의입니다. `expert0` 등에 MC 평균을 넣어 MC 결합 결과를
복원하면 clipping 등의 비선형 처리 때문에 다른 값이 될 수 있습니다.
두 번째 표의 MC 결과에는 `tools/replay_revision.py`로 저장된 draw를
재생하는 경로를 사용합니다. 재계산 도구가 과거 checkpoint 선택 이력이나
게이트의 학습 자료를 추정해 주는 것은 아닙니다.

## 논문 수정 방향

우선 기존 Bi-LSTM 구조와 조건별 전문가 결합 아이디어를 중심으로 수정
가능 범위를 검토합니다. Transformer 결과는 더 개선할 수 있는 방향으로
함께 제시하며, 주모델 변경 여부와 논문 반영 범위는 교수님과 상의하여
결정할 항목입니다.

이번 결과는 기존 방법을 보완하여 현재 자료에서 성능 개선 가능성을
살펴본 추가 실험입니다. 학습·평가자료의 원본 연관성을 고려하여, 새로운
독립 시편에 대한 성능은 후속 검증 항목으로 두었습니다.
