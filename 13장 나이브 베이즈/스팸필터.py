from typing import Set
from typing import NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict


def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)


print(tokenize("Data Science is science"))


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k

        self.tokens: Set[str] = set()  # 메시지에 나타나는 모든 토큰들 저장
        self.token_spam_counts: Dict[str, int] = defaultdict(int)  # 토크이 스팸에 나타난 빈도
        self.token_ham_counts: Dict[str, int] = defaultdict(int)  # 토큰이 햄에 나타난 빈도
        self.spam_messages = self.ham_messages = 0  # 각 메시지 종류별 메시지 수

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


# P(A|S)= 0.1
# P(~A|S) = 0.9
#
#
# P(A|~S)

# 학습데이터 만들기
messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]
model = NaiveBayesClassifier(k=0.5)
model.train(messages)

# 훈련데이터 모델 확인 : assert  print로 교체해서 테스트
assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}
# 메시지 “hello spam”의 스팸여부 판단 예 : 수작업으로 구하고, 구현 값과 비교
text = "hello spam"
probs_if_spam = [(1 + 0.5) / (1 + 2 * 0.5),  # "spam" (present)
                 1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham" (not present)
                 1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
                 (0 + 0.5) / (1 + 2 * 0.5)  # "hello" (present)
                 ]
probs_if_ham = [(0 + 0.5) / (2 + 2 * 0.5),  # "spam" (present)
                1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham" (not present)
                1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
                (1 + 0.5) / (2 + 2 * 0.5),  # "hello" (present)
                ]
p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))
# print로 교체해서 비교: Should be about 0.83
print(model.predict(text))
print(p_if_spam / (p_if_spam + p_if_ham))
