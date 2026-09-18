import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_processing"))
import threading
from database_service import DatabaseService


def test_concurrent_writes_do_not_lose_data():
    with DatabaseService("carivix_api.db") as setup_db:
        setup_db.db.drop_table("concurrency_test", if_exists=True)
        setup_db.db.create_table("concurrency_test", {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "worker": "INTEGER",
            "value": "TEXT",
        })

    errors = []

    def write_batch(n):
        try:
            with DatabaseService("carivix_api.db") as db:
                db.write("concurrency_test", [{"worker": n, "value": f"item_{n}"}])
        except Exception as e:
            errors.append((n, str(e)))

    threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    with DatabaseService("carivix_api.db") as db:
        rows = db.read("concurrency_test")
    assert len(rows) == 10