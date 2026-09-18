from db_crud import DatabaseManager


def make_db():
    db = DatabaseManager(":memory:")
    db.create_table("items", {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT NOT NULL",
        "value": "REAL",
    })
    return db


def test_insert_and_read():
    db = make_db()
    db.insert("items", {"name": "apple", "value": 1.5})
    rows = db.read("items")
    assert len(rows) == 1
    assert rows[0]["name"] == "apple"


def test_insert_many():
    db = make_db()
    db.insert_many("items", [
        {"name": "apple", "value": 1.5},
        {"name": "banana", "value": 2.0},
    ])
    assert db.count("items") == 2


def test_update_changes_matching_rows():
    db = make_db()
    db.insert("items", {"name": "apple", "value": 1.5})
    updated = db.update("items", {"value": 3.0}, where={"name": "apple"})
    row = db.read_one("items", where={"name": "apple"})
    assert updated == 1
    assert row["value"] == 3.0


def test_delete_removes_matching_rows():
    db = make_db()
    db.insert("items", {"name": "apple", "value": 1.5})
    deleted = db.delete("items", where={"name": "apple"})
    assert deleted == 1
    assert db.count("items") == 0


def test_count_with_filter():
    db = make_db()
    db.insert_many("items", [
        {"name": "apple", "value": 1.5},
        {"name": "apple", "value": 2.0},
        {"name": "banana", "value": 3.0},
    ])
    assert db.count("items", where={"name": "apple"}) == 2