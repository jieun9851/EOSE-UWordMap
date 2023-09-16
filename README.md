# 어휘의미망을 이용한 주제 분류 및 감성 표현 영역 추출 모델 (수정중)

## Paper: [A Model for Topic Classification and Extraction of Sentimental Expression using a Lexical Semantic Network](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11495798)

<br/>
<br/>

본 논문의 제안 모델은 어절 단위 감성 표현 영역 추출 모델이다. 제안 모델은 주제별 사전으로 예측한 문장의 주제를 모델의 자질로 사용한다. 주제별 사전은 학습 단계 초기에 구축되며, 학습 모듈이 학습 말뭉치에서 주제별 단어를 수집하고 어휘의미망의 상하관계를 이용해 주제별 단어를 확장한다. 제안 모델의 구조는 형태소 분석된 문장을 입력으로 사용하는 UBERT 모델에 주제 분류와 감성 표현 영역을 예측하는 레이어를 추가한 것이다. 

![image](https://github.com/jieun9851/EOSE-UWordMap/assets/57825347/ce307841-f8dd-4bd6-a97a-e4047c8d24e2)

<br/>
<br/>

### 데이터
데이터는 모두의 말뭉치 - 감성 분석 데이터를 사용했다. 기존의 데이터들은 짧은 문장으로 구성되어 있어 문장 내 정보가 적어 주제나 감성 표현 영역을 알기 쉽지 않았다. 그래서 본 실험에서는 동일 문서에서 100자 미만인 문장을 연결하여 긴 문장으로 재구성했다.
- total dataset: 25,482개 => 7,580개 
- train dataset: 6,064개, test dataset: 1,516개
- train_span_ud_nv_ej.txt, test_span_ud_nv_ej.txt
- 

<br/>
<br/>

### 주제별 사전
본 논문의 알고리즘으로 구축한 주제별 사전을 공개합니다. UWordMap 기반으로 제작된 사전입니다. 


<br/>
<br/>

### 모델

소스코드는 Google-BERT의 오픈 소스를 기반으로 제작했으며, 논문 속 UBERT 관련 weight는 연구실 내에서 사용하는 모델이므로 첨부하지 못했다.
본 소스코드는 run 부분이 저자가 수정한 코드 부분이다. 
소스코드를 이용해서 실험을 진행하고 싶다면 본인이 제작한 weight를 교체해서 사용하면 될 듯 하다.


모델에는 크게 4가지 버전이 있다.
run_span_finetuning.py
run_span_mtl_finetuning.py
run_span_ud_mtl_finetuning.py
run_span_up_down_finetuning.py
run_span_up_nv_down_finetuning_ej.py


1. UBERT 토큰 단위 ner version
2. UBERT 토큰 단위 span version
3. UBERT 어절 단위 ner version
4. UBERT 어절 단위 span version


### 실행 방법

