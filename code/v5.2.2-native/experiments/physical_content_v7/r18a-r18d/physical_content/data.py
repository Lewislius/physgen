"""Training view restricted to audited rows whose frozen Qwen evidence exists."""
from physical_scaling.data import Data
from .common import FEATURE_CACHE
from .evidence import find_input


class TrainingData(Data):
    """Use every train video that has complete cached first-image/request evidence.

    The manifest remains 3857 records, but only rows with all required immutable
    pieces (positive and null Qwen words/native targets and
    first-image/request evidence) can form a valid training condition.  No missing row is silently
    replaced by a null condition.  When the user completes the missing caches,
    the full split is used automatically without changing code.
    """

    def __init__(self, cfg):
        super().__init__(cfg, 'train')
        rows = []
        for row in self.rows:
            complete = True
            for name in (f'{row["id"]}_text', f'{row["id"]}_null'):
                if not (FEATURE_CACHE / 'words' / f'{name}.pt').is_file():
                    complete = False
                    break
                if not (FEATURE_CACHE / 'native_words' / f'{name}.pt').is_file():
                    complete = False
                    break
                try:
                    find_input(name)
                except FileNotFoundError:
                    complete = False
                    break
            if complete:
                rows.append(row)
        if not rows:
            raise ValueError('No train row has complete frozen Qwen evidence')
        self.rows = rows
        self.by_id = {row['id']: row for row in rows}
