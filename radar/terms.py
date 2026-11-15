"""Извлечение терминов-кандидатов (1–3-словных фраз) из текстов."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

STOP = set("""
a about above across after again against all almost along already also although always am among an
and another any are around as at away be became because become becomes been before being below
between beyond both but by can cannot could did do does doing done down due during each either else
enough especially etc even ever every few for from further get gets given gives go had has have
having he her here hers him his how however i if in into is it its itself just keep kept known last
least less let like likely made make makes many may me might more most much must my near nearly need
needs neither never no nor not now of off often on once one only onto or other others otherwise our
ours out over own per perhaps quite rather really same see seem seems several shall she should since
so some something still such than that the their theirs them themselves then there therefore these
they this those though through throughout thus to together too toward towards under unless until up
upon us use used uses using usually very via was we well were what whatever when where whether which
while who whom whose why will with within without would yet you your yours we've it's i'm don't
doesn't isn't aren't can't won't show ask tell launch hn yc paper propose proposed proposes present
presents presented introduce introduces introduced novel new result results show shows shown
demonstrate demonstrates demonstrated achieve achieves achieved achieving outperform outperforms
outperforming existing recent recently various different several two three four five first second
third based significant significantly effective effectively efficient efficiently improve improves
improved improving improvement improvements enable enables enabling allow allows provide provides
providing work works experiment experiments experimental extensive comprehensive state art sota
issue issues key important crucial critical potential promising remains remain challenge challenges
challenging limited limitation limitations specifically particular particularly furthermore moreover
additionally overall finally respectively including include includes code available github https
http www com org io html pdf arxiv preprint et al ie eg vs versus whereas whose upon order terms
term large small high low higher lower better best good great simple simply directly large-scale
real-world analyze analyzed evaluate evaluated learning-based part parts case cases year years day
days paper's lead leads leading led consider considered considering term terms aim aims goal goals
across range question questions answer answers called named simple free
""".split())

STOP |= set("""january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday today yesterday week weeks month
months""".split())

# Общие слова: допустимы внутри фразы («world model»), но фраза только из них — шум
GENERIC = set("""
approach approaches method methods framework frameworks task tasks problem problems study studies benchmark benchmarks dataset datasets model models data performance accuracy training trained train analysis evaluation evaluations level levels way ways number numbers set sets type types form forms time times scale quality tool tools app apps open source
""".split())

# Слова, которые сами по себе (униграммами) почти всегда шум
WEAK_UNIGRAM = GENERIC | set("""
system systems network networks method neural deep learning machine human information process
processes function functions structure structures based effect effects role roles signal signals
control image images video language text user users design feature features value values space
graph graphs agent agents state states series quality cost costs scale world people things thing
using build building built project projects library framework release version update support
""".split())

_TOKEN = re.compile(r"[a-z][a-z0-9+]*(?:\.[0-9]+)?|[0-9]+[a-z][a-z0-9]*")
_SPLIT = re.compile(r"[.,;:!?()\[\]{}\"'“”‘’«»<>|=*#@~^`$%&\\]+|\s-\s|\s–\s|\s—\s")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NUM_ONLY = re.compile(r"^[0-9.]+$")


def _norm_token(t: str) -> str:
    """Грубая лемматизация: убираем множественное число."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is", "os", "as")):
        return t[:-1]
    return t


def _prep(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _URL.sub(" ", text)
    text = re.sub(r"[-_/]", " ", text)
    return text


def extract(text: str, tags: list[str] | None = None, max_n: int = 3) -> dict[str, str]:
    """Возвращает {нормализованный_термин: исходное написание} для одного документа."""
    out: dict[str, str] = {}
    for chunk in _SPLIT.split(_prep(text)):
        raw = _TOKEN.findall(chunk)
        toks = [(_norm_token(t), t) for t in raw]
        n_tok = len(toks)
        for i in range(n_tok):
            for n in range(1, max_n + 1):
                if i + n > n_tok:
                    break
                gram = toks[i:i + n]
                norms = [g[0] for g in gram]
                if any(w in STOP or len(w) < 2 or _NUM_ONLY.match(w) for w in norms):
                    break  # дальше расширять фразу через стоп-слово нет смысла
                if n == 1 and (len(norms[0]) < 4 or norms[0] in WEAK_UNIGRAM):
                    continue
                if n > 1 and all(w in GENERIC for w in norms):
                    continue
                key = " ".join(norms)
                out.setdefault(key, " ".join(g[1] for g in gram))
    for t in tags or []:
        t = _prep(t).strip()
        toks = _TOKEN.findall(t)
        if 1 <= len(toks) <= 3 and not all(w in STOP for w in toks):
            key = " ".join(_norm_token(w) for w in toks)
            if len(toks) > 1 or (len(key) >= 4 and key not in WEAK_UNIGRAM and key not in STOP):
                out.setdefault(key, " ".join(toks))
    return out


try:  # частотность английских слов: отсеивает обычные слова-униграммы («journey», «invite»)
    from wordfreq import zipf_frequency as _zipf
except ImportError:  # pragma: no cover
    _zipf = None


def is_common_word(word: str, threshold: float = 3.0) -> bool:
    return bool(_zipf) and _zipf(word, "en") >= threshold


class WeekIndex:
    """Частоты терминов за одну неделю: документная частота по источникам + обратный индекс."""

    def __init__(self, docs):
        self.docs = docs
        self.n_docs: Counter = Counter(d.source for d in docs)
        self.df: dict[str, Counter] = defaultdict(Counter)
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.title_df: Counter = Counter()  # в скольких документах термин стоит в заголовке
        self.word_df: Counter = Counter()   # частота отдельных слов — для «специфичности» фраз
        surface: dict[str, Counter] = defaultdict(Counter)
        for i, d in enumerate(docs):
            terms = extract(d.text, d.tags)
            for key, surf in terms.items():
                self.df[key][d.source] += 1
                self.postings[key].append(i)
                surface[key][surf] += 1
            # теги GitHub/HF — тоже «заголовочный» уровень: автор сам назвал так свой проект
            self.title_df.update(set(extract(d.title, d.tags)) & set(terms))
            self.word_df.update({_norm_token(t) for t in _TOKEN.findall(_prep(d.text))})
        self.surface = {k: c.most_common(1)[0][0] for k, c in surface.items()}

    def title_share(self, term: str) -> float:
        n = self.total(term)
        return self.title_df.get(term, 0) / n if n else 0.0

    def specificity(self, term: str) -> float:
        """Какая доля документов с самым редким словом фразы содержит всю фразу.
        «world model» — высокая, случайное «single machine» — почти ноль."""
        words = [w for w in term.split() if w not in GENERIC]
        if len(term.split()) < 2 or not words:
            return 1.0
        return self.total(term) / max(min(self.word_df.get(w, 1) for w in words), 1)

    def total(self, term: str) -> int:
        return sum(self.df.get(term, {}).values())

    def to_history(self, min_df: int = 2) -> dict:
        return {"n_docs": dict(self.n_docs),
                "terms": {t: dict(c) for t, c in self.df.items() if sum(c.values()) >= min_df}}
