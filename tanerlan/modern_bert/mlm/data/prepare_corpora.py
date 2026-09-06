"""Подготовка локальных корпусов для MLM в mlm-data/<bucket>/{train,validation}.parquet.

Бакеты и бюджеты в символах подобраны под смесь uz-lat 45 / uz-cyr 25 /
ru 20 / en 5 (оставшиеся 5% — синтетический code-switching, он собирается
на лету через CodeSwitchingDataConfig и здесь не готовится):

  uz-latn          латинский узбекский из uz-crawl (news + telegram_blogs)
  uz-cyrl-books    кириллица из uz-books (OCR, с фильтром шумных чанков)
  uz-cyrl-translit латинский crawl, транслитерированный в кириллицу
                   (чистый соцмедиа-текст без OCR-шума; документы не
                   пересекаются с uz-latn — берутся из хвоста того же потока)
  ru-wiki          wikimedia/wikipedia 20231101.ru
  ru-news          IlyaGusev/gazeta (новости, регистр ближе к соцмедиа)
  en-wiki          wikimedia/wikipedia 20231101.en

Документы режутся на чанки по границам абзацев (иначе книга целиком уйдёт
в один пример и обрежется при токенизации), каждый 50-й чанк — в validation.
Чанки короче --min-chunk-chars (после strip) выбрасываются: хвостовые пустые
абзацы, одинокие заголовки и списки категорий из wiki дают примеры из пары
токенов, в которых MLM-коллатору нечего маскировать (loss на таком батче —
0/0 = NaN). Готовые бакеты при повторном запуске пропускаются.
"""

from typing import Literal
from types import SimpleNamespace
from collections.abc import Iterator

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import rich_click as click
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

click.rich_click.USE_MARKDOWN = True
console = Console()

# --- бюджеты в символах (scale=1.0), пропорции 45/25/20/5 без учёта CS ----------

BUDGETS = {
    "uz-latn": 600_000_000,
    "uz-cyrl-books": 170_000_000,
    "uz-cyrl-translit": 170_000_000,
    "ru-wiki": 180_000_000,
    "ru-news": 90_000_000,
    "en-wiki": 70_000_000,
}

# --- определение письменности (как в старом tokenizer/corpus.py) ----------------

_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_UZ_LATIN_MARKS = re.compile(
    r"(?i)[oq]ʻ|\bq\w|\w(ning|lar|lari|dagi|da|ga|ni|di|gan|moq|siz|lik|chi)\b"
    r"|\b(va|bu|bilan|uchun|ham|bir|deb|edi|emas|haqida|boʻlib)\b"
)
_EN_MARKS = re.compile(r"(?i)\b(the|and|of|to|in|is|for|that|with|are)\b")
_UZ_CYRILLIC_MARKS = re.compile(
    r"(?i)[ўғқҳ]|\w(нинг|лар|лари|даги|да|га|ни|ди|ган|моқ|сиз|лик|чи)\b"
    r"|\b(ва|бу|билан|учун|ҳам|бир|деб|эди|эмас|ҳақида)\b"
)
_RU_MARKS = re.compile(r"(?i)[ыщ]|\b(и|в|не|на|что|с|по|для|как|это|к|но|от|из)\b")


def classify_script(text: str) -> Literal["latin", "cyrillic"] | None:
    """latin / cyrillic для узбекского текста, None для английского, русского и смеси."""
    latin = len(_LATIN.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    total = latin + cyrillic
    if total < 20:
        return None
    if latin / total >= 0.8:
        uz, other = len(_UZ_LATIN_MARKS.findall(text)), len(_EN_MARKS.findall(text))
        return "latin" if uz > other else None
    if cyrillic / total >= 0.8:
        uz, other = len(_UZ_CYRILLIC_MARKS.findall(text)), len(_RU_MARKS.findall(text))
        return "cyrillic" if uz > other else None
    return None


def split_chunks(text: str, max_chars: int) -> Iterator[str]:
    """Режет документ на куски не длиннее max_chars по границам абзацев."""
    if len(text) <= max_chars:
        yield text
        return
    buffer: list[str] = []
    size = 0
    for paragraph in text.split("\n"):
        if size + len(paragraph) > max_chars and buffer:
            yield "\n".join(buffer)
            buffer, size = [], 0
        buffer.append(paragraph)
        size += len(paragraph) + 1
    if buffer:
        yield "\n".join(buffer)


# --- транслитерация лат → кир ---------------------------------------------------

# после prepare_input апострофы унифицированы: oʻ/gʻ через U+02BB, tutuq — U+02BC
_LAT2CYR_MULTI = [
    ("oʻ", "ў"), ("Oʻ", "Ў"),
    ("gʻ", "ғ"), ("Gʻ", "Ғ"),
    ("sh", "ш"), ("Sh", "Ш"), ("SH", "Ш"),
    ("ch", "ч"), ("Ch", "Ч"), ("CH", "Ч"),
    ("yo", "ё"), ("Yo", "Ё"), ("YO", "Ё"),
    ("yu", "ю"), ("Yu", "Ю"), ("YU", "Ю"),
    ("ya", "я"), ("Ya", "Я"), ("YA", "Я"),
    ("ts", "ц"), ("Ts", "Ц"), ("TS", "Ц"),
]
_LAT2CYR_SINGLE = str.maketrans(
    "abdefghijklmnopqrstuvxyzABDEFGHIJKLMNOPQRSTUVXYZʼ",
    "абдефгҳижклмнопқрстувхйзАБДЕФГҲИЖКЛМНОПҚРСТУВХЙЗъ",
)
_WORD_INITIAL_E = re.compile(r"\b[eE]")


def uz_lat_to_cyr(text: str) -> str:
    """Приближённая детерминированная транслитерация узбекской латиницы в кириллицу.

    Систематические огрехи (e/э не в начале слова, заимствования с ц) редки
    и, в отличие от OCR-шума, не учат модель случайному мусору.
    """
    text = prepare_input(text)
    for lat, cyr in _LAT2CYR_MULTI:
        text = text.replace(lat, cyr)
    text = _WORD_INITIAL_E.sub(lambda m: "Э" if m.group(0) == "E" else "э", text)
    return text.translate(_LAT2CYR_SINGLE)


# --- качество OCR-чанков --------------------------------------------------------

_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def looks_clean(chunk: str) -> bool:
    """Отсекает OCR-мусор: мало букв, рваные однобуквенные обломки, аномальные слова."""
    words = chunk.split()
    if len(words) < 20:
        return False
    letters = len(_LETTERS.findall(chunk))
    if letters / len(chunk) < 0.6:
        return False
    short = sum(1 for word in words if len(_LETTERS.findall(word)) <= 1)
    if short / len(words) > 0.2:
        return False
    mean_len = sum(len(w) for w in words) / len(words)
    return 3.0 <= mean_len <= 14.0


# --- запись parquet -------------------------------------------------------------

_SCHEMA = pa.schema([("text", pa.string())])


class BucketWriter:
    """Пишет чанки в train/validation.parquet, каждый val_every-й чанк — в validation.

    Чанки короче min_chars значимых символов не пишутся и не участвуют в
    раскладке по сплитам; их число копится в dropped.
    """

    def __init__(self, directory: Path, val_every: int, min_chars: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # пишем во .tmp и переименовываем в close(): прерванный бакет
        # не выглядит готовым и пересобирается при следующем запуске
        self.directory = directory
        self.train = pq.ParquetWriter(directory / "train.parquet.tmp", _SCHEMA)
        self.val = pq.ParquetWriter(directory / "validation.parquet.tmp", _SCHEMA)
        self.val_every = val_every
        self.min_chars = min_chars
        self.buffers: dict[str, list[str]] = {"train": [], "val": []}
        self.chars = 0
        self.chunks = 0
        self.dropped = 0

    def add(self, chunk: str) -> None:
        if len(chunk.strip()) < self.min_chars:
            self.dropped += 1
            return
        subset = "val" if self.chunks % self.val_every == self.val_every - 1 else "train"
        self.buffers[subset].append(chunk)
        self.chars += len(chunk)
        self.chunks += 1
        if len(self.buffers[subset]) >= 5_000:
            self._flush(subset)

    def _flush(self, subset: str) -> None:
        if not self.buffers[subset]:
            return
        writer = self.train if subset == "train" else self.val
        writer.write_table(pa.table({"text": self.buffers[subset]}, schema=_SCHEMA))
        self.buffers[subset] = []

    def close(self) -> None:
        for subset in self.buffers:
            self._flush(subset)
        self.train.close()
        self.val.close()
        for name in ("train.parquet", "validation.parquet"):
            (self.directory / f"{name}.tmp").rename(self.directory / name)


# --- сборка бакетов -------------------------------------------------------------


def _interleave(*iterators: Iterator[str]) -> Iterator[str]:
    """Чередует источники, пока все не исчерпаются: news и telegram идут вперемешку."""
    pending = list(iterators)
    while pending:
        for iterator in list(pending):
            try:
                yield next(iterator)
            except StopIteration:
                pending.remove(iterator)


def _stream_texts(
    repo: str, split: str, config: str | None = None, text_colname: str = "text"
) -> Iterator[str]:
    stream = load_dataset(repo, config, split=split, streaming=True)
    for record in stream:
        text = record.get(text_colname)
        if text and len(text) >= 200:
            yield text


def _bucket_done(directory: Path) -> bool:
    return (directory / "train.parquet").exists() and (
        directory / "validation.parquet"
    ).exists()


BucketStats = tuple[int, int]  # (символов записано, чанков отброшено)


def prepare_uz_crawl(
    args: SimpleNamespace, budgets: dict[str, int]
) -> dict[str, BucketStats]:
    """Один проход по crawl: сперва заполняется uz-latn, хвост потока — в транслитерацию."""
    latn_dir = args.output / "uz-latn"
    translit_dir = args.output / "uz-cyrl-translit"
    if _bucket_done(latn_dir) and _bucket_done(translit_dir):
        console.print("[dim]uz-latn и uz-cyrl-translit уже готовы, пропускаю[/]")
        return {}

    latn = BucketWriter(latn_dir, args.val_every, args.min_chunk_chars)
    translit = BucketWriter(translit_dir, args.val_every, args.min_chunk_chars)
    stream = _interleave(
        _stream_texts("tahrirchi/uz-crawl", "news"),
        _stream_texts("tahrirchi/uz-crawl", "telegram_blogs"),
    )
    with console.status("uz-crawl → uz-latn + uz-cyrl-translit") as status:
        for text in stream:
            if classify_script(text) != "latin":
                continue
            for chunk in split_chunks(text, args.chunk_chars):
                if latn.chars < budgets["uz-latn"]:
                    latn.add(chunk)
                elif translit.chars < budgets["uz-cyrl-translit"]:
                    translit.add(uz_lat_to_cyr(chunk))
            if translit.chars >= budgets["uz-cyrl-translit"]:
                break
            status.update(
                f"uz-latn {latn.chars / 1e6:.0f}M / {budgets['uz-latn'] / 1e6:.0f}M · "
                f"translit {translit.chars / 1e6:.0f}M / {budgets['uz-cyrl-translit'] / 1e6:.0f}M"
            )
    latn.close()
    translit.close()
    return {
        "uz-latn": (latn.chars, latn.dropped),
        "uz-cyrl-translit": (translit.chars, translit.dropped),
    }


def prepare_simple_bucket(
    args: SimpleNamespace,
    name: str,
    budget: int,
    texts: Iterator[str],
    script: Literal["latin", "cyrillic"] | None = None,
    quality: bool = False,
) -> dict[str, BucketStats]:
    directory = args.output / name
    if _bucket_done(directory):
        console.print(f"[dim]{name} уже готов, пропускаю[/]")
        return {}
    writer = BucketWriter(directory, args.val_every, args.min_chunk_chars)
    with console.status(name) as status:
        for text in texts:
            if script is not None and classify_script(text) != script:
                continue
            for chunk in split_chunks(text, args.chunk_chars):
                if quality and not looks_clean(chunk):
                    continue
                writer.add(chunk)
            if writer.chars >= budget:
                break
            status.update(f"{name} {writer.chars / 1e6:.0f}M / {budget / 1e6:.0f}M")
    writer.close()
    return {name: (writer.chars, writer.dropped)}


def run(args: SimpleNamespace) -> None:
    budgets = {name: int(budget * args.scale) for name, budget in BUDGETS.items()}
    args.output.mkdir(parents=True, exist_ok=True)

    written: dict[str, BucketStats] = {}
    written |= prepare_uz_crawl(args, budgets)
    written |= prepare_simple_bucket(
        args,
        "uz-cyrl-books",
        budgets["uz-cyrl-books"],
        _stream_texts("tahrirchi/uz-books", "original"),
        script="cyrillic",
        quality=True,
    )
    written |= prepare_simple_bucket(
        args,
        "ru-wiki",
        budgets["ru-wiki"],
        _stream_texts("wikimedia/wikipedia", "train", config="20231101.ru"),
    )
    written |= prepare_simple_bucket(
        args,
        "ru-news",
        budgets["ru-news"],
        _stream_texts("IlyaGusev/gazeta", "train"),
    )
    written |= prepare_simple_bucket(
        args,
        "en-wiki",
        budgets["en-wiki"],
        _stream_texts("wikimedia/wikipedia", "train", config="20231101.en"),
    )

    table = Table(title=f"Корпуса в {args.output}")
    table.add_column("бакет")
    table.add_column("символов", justify="right")
    table.add_column("отброшено чанков", justify="right")
    for name, (chars, dropped) in written.items():
        table.add_row(name, f"{chars:,}", f"{dropped:,}")
    console.print(table)


@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("mlm-data"),
    show_default=True,
    help="Куда сложить бакеты (по подпапке на бакет).",
)
@click.option(
    "--scale",
    type=click.FloatRange(min=0.01),
    default=1.0,
    show_default=True,
    help="Множитель символьных бюджетов всех бакетов.",
)
@click.option(
    "--chunk-chars",
    type=click.IntRange(min=500),
    default=4_000,
    show_default=True,
    help="Максимальная длина чанка; документы режутся по абзацам.",
)
@click.option(
    "--min-chunk-chars",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Чанки короче этого (после strip) выбрасываются; 0 отключает фильтр.",
)
@click.option(
    "--val-every",
    type=click.IntRange(min=2),
    default=50,
    show_default=True,
    help="Каждый N-й чанк уходит в validation.",
)
def main(**options: object) -> None:
    """Готовит локальные MLM-корпуса: uz (лат/кир/транслит), ru (wiki+news), en (wiki)."""
    try:
        run(SimpleNamespace(**options))
    except (OSError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
