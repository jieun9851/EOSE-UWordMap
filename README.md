# EOSE-UWordMap

## [A Model for Topic Classification and Extraction of Sentimental Expression using a Lexical Semantic Network](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11495798).

본 모델은 Google-BERT 코드를 기반으로 제작했으며, 논문 속 UBERT 관련 weight는 연구실 내에서 사용하는 모델이므로 첨부하지 못했다.
본 소스코드는 run 부분을 수정한 부분을 첨부한 것이다. 
본인이 제작한 weight를 사용해서 돌리고 싶은 분들은 기존의 bert 코드를 다운 받은 후 해당 run 부분만을 교체해서 사용하면 될 듯 하다.

모델에는 크게 4가지 버전이 있다.

1. UBERT 토큰 단위 ner version
2. UBERT 토큰 단위 span version
3. UBERT 어절 단위 ner version
4. UBERT 어절 단위 span version
