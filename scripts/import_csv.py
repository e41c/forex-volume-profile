# scripts/import_csv.py
"""
Drop your year folders into data/raw/ then run:
    python scripts/import_csv.py

Safe to re-run — already imported years are skipped.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.csv_provider import import_all_csvs
from src.data.db_manager   import get_stats
from src.utils.logger       import get_logger

log = get_logger("importer")

if __name__ == "__main__":
    log.info("=" * 50)
    log.info("GOBLIN CSV IMPORTER STARTING 🐲")
    log.info("=" * 50)

    import_all_csvs(raw_dir="data/raw", skip_existing=True)

    log.info("\n📊 Database summary:")
    stats = get_stats()
    if stats.empty:
        log.warning("No data in database yet")
    else:
        print(stats.to_string(index=False))