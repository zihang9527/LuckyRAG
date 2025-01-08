# @Author : Shichao Song
# @Email  : song.shichao@outlook.com

from typing import Callable
import evaluate
import jieba
from logger import logger

def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(repr(e))
    return wrapper


@catch_all_exceptions
def bleu_score(
    continuation: str,
    reference: str,
    with_penalty = False
) -> float:
    f = lambda text: list(jieba.cut(text))
    bleu = evaluate.load('src/.cache/huggingface/bleu')
    results = bleu.compute(predictions=[continuation], references=[[reference]], tokenizer=f)
    
    bleu_avg = results['bleu']
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    bleu3 = results['precisions'][2]
    bleu4 = results['precisions'][3]
    brevity_penalty = results['brevity_penalty']

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return 0.0 if brevity_penalty==0 else bleu_avg/brevity_penalty, bleu1, bleu2, bleu3, bleu4


@catch_all_exceptions
def rougeL_score(
    continuation: str,
    reference: str
) -> float:
    f = lambda text: list(jieba.cut(text))
    rouge = evaluate.load('src/.cache/huggingface/rouge')
    results = rouge.compute(predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']
    return score

