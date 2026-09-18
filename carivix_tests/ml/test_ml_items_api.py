"""
ML Items CRUD API Tests
=========================

Service under test : http://127.0.0.1:8002  (ML module/fastapi_main.py)
Playwright fixture : ``items_api`` (APIRequestContext) from conftest.py

fastapi_main.py exposes a simple in-memory Items API with:
  GET  /items              – list all items
  GET  /items/{id}         – get single item
  POST /items              – create item  (returns 201)
  PUT  /items/{id}         – update item

On startup the service pre-populates two sample items:
  id=1 – Sample A (price 1.23)
  id=2 – Sample B (price 4.56)

Run:
    pytest carivix_tests/ml/test_ml_items_api.py -v
    pytest -m items -v
"""

from __future__ import annotations

import pytest
from playwright.sync_api import APIRequestContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_item(items_api: APIRequestContext, name: str, price: float, description: str = "") -> dict:
    """POST /items and return the created item body."""
    payload: dict = {"name": name, "price": price}
    if description:
        payload["description"] = description
    resp = items_api.post("/items", data=payload)
    assert resp.status == 201, f"Create item failed: {resp.status} {resp.text()}"
    return resp.json()


# ===========================================================================
# List items (GET /items)
# ===========================================================================

@pytest.mark.items
@pytest.mark.smoke
def test_items_list_returns_200(items_api: APIRequestContext):
    """GET /items returns HTTP 200."""
    response = items_api.get("/items")
    assert response.status == 200, f"Expected 200, got {response.status}"


@pytest.mark.items
@pytest.mark.smoke
def test_items_list_returns_array(items_api: APIRequestContext):
    """GET /items body is a JSON array."""
    body = items_api.get("/items").json()
    assert isinstance(body, list), f"Expected list, got {type(body)}"


@pytest.mark.items
def test_items_list_has_seeded_items(items_api: APIRequestContext):
    """Startup populates at least 2 sample items (Sample A and Sample B)."""
    items = items_api.get("/items").json()
    assert len(items) >= 2, f"Expected ≥ 2 seeded items, got {len(items)}"


@pytest.mark.items
def test_items_list_each_item_has_required_fields(items_api: APIRequestContext):
    """Every item in the list has 'id', 'name', 'price'."""
    items = items_api.get("/items").json()
    for item in items:
        for field in ("id", "name", "price"):
            assert field in item, f"Item missing '{field}': {item}"


# ===========================================================================
# Get single item (GET /items/{id})
# ===========================================================================

@pytest.mark.items
def test_items_get_existing_item_returns_200(items_api: APIRequestContext):
    """GET /items/1 returns HTTP 200 for the first seeded item."""
    response = items_api.get("/items/1")
    assert response.status == 200, f"Expected 200, got {response.status}"


@pytest.mark.items
def test_items_get_item_body_structure(items_api: APIRequestContext):
    """GET /items/1 body contains id, name, price fields."""
    body = items_api.get("/items/1").json()
    assert body["id"] == 1
    assert isinstance(body["name"], str)
    assert isinstance(body["price"], float)


@pytest.mark.items
def test_items_get_nonexistent_item_returns_404(items_api: APIRequestContext):
    """GET /items/99999 returns HTTP 404 for an item that does not exist."""
    response = items_api.get("/items/99999", fail_on_status_code=False)
    assert response.status == 404, f"Expected 404, got {response.status}"


@pytest.mark.items
def test_items_get_404_response_has_detail(items_api: APIRequestContext):
    """404 response for non-existent item includes a 'detail' error message."""
    body = items_api.get("/items/99999", fail_on_status_code=False).json()
    assert "detail" in body, f"Missing 'detail' in 404 body: {body}"


# ===========================================================================
# Create item (POST /items)
# ===========================================================================

@pytest.mark.items
def test_items_create_returns_201(items_api: APIRequestContext):
    """POST /items with valid payload returns HTTP 201 Created."""
    response = items_api.post("/items", data={"name": "Test Widget", "price": 12.99})
    assert response.status == 201, f"Expected 201, got {response.status}"


@pytest.mark.items
def test_items_create_returns_full_item(items_api: APIRequestContext):
    """POST /items response includes the created item with an auto-assigned id."""
    body = items_api.post("/items", data={"name": "New Gadget", "price": 29.99}).json()
    assert "id" in body and body["id"] is not None
    assert body["name"] == "New Gadget"
    assert body["price"] == 29.99


@pytest.mark.items
def test_items_create_with_description(items_api: APIRequestContext):
    """POST /items with optional 'description' field stores and returns it."""
    body = items_api.post(
        "/items",
        data={"name": "Described Item", "price": 5.0, "description": "A useful thing"},
    ).json()
    assert body["description"] == "A useful thing"


@pytest.mark.items
def test_items_create_item_appears_in_list(items_api: APIRequestContext):
    """Item created via POST /items is subsequently visible in GET /items."""
    unique_name = f"Unique_Item_pytest"
    created = _create_item(items_api, unique_name, 7.77)
    item_id = created["id"]

    items = items_api.get("/items").json()
    ids = [i["id"] for i in items]
    assert item_id in ids, f"Created item id={item_id} not found in list"


@pytest.mark.items
def test_items_create_missing_name_returns_422(items_api: APIRequestContext):
    """POST /items without 'name' (required field) → 422 Unprocessable Entity."""
    response = items_api.post("/items", data={"price": 5.0}, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422, got {response.status}"


@pytest.mark.items
def test_items_create_missing_price_returns_422(items_api: APIRequestContext):
    """POST /items without 'price' (required field) → 422."""
    response = items_api.post("/items", data={"name": "No Price"}, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.items
def test_items_create_increments_id(items_api: APIRequestContext):
    """Successive POST /items calls produce monotonically increasing IDs."""
    i1 = _create_item(items_api, "Item A", 1.0)
    i2 = _create_item(items_api, "Item B", 2.0)
    assert i2["id"] > i1["id"], f"id not incrementing: {i1['id']} → {i2['id']}"


# ===========================================================================
# Update item (PUT /items/{id})
# ===========================================================================

@pytest.mark.items
def test_items_update_existing_item_returns_200(items_api: APIRequestContext):
    """PUT /items/1 with valid payload returns HTTP 200."""
    response = items_api.put("/items/1", data={"name": "Sample A Updated", "price": 2.46})
    assert response.status == 200, f"Expected 200, got {response.status}: {response.text()}"


@pytest.mark.items
def test_items_update_returns_updated_body(items_api: APIRequestContext):
    """PUT /items/1 response body reflects the new values."""
    body = items_api.put(
        "/items/1", data={"name": "Modified A", "price": 99.0}
    ).json()
    assert body["id"] == 1
    assert body["name"] == "Modified A"
    assert body["price"] == 99.0


@pytest.mark.items
def test_items_update_nonexistent_item_returns_404(items_api: APIRequestContext):
    """PUT /items/99999 → 404 when item does not exist."""
    response = items_api.put(
        "/items/99999",
        data={"name": "Ghost", "price": 0.0},
        fail_on_status_code=False,
    )
    assert response.status == 404


@pytest.mark.items
def test_items_update_missing_name_returns_422(items_api: APIRequestContext):
    """PUT /items/1 without 'name' → 422 Unprocessable Entity."""
    response = items_api.put("/items/1", data={"price": 5.0}, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.items
def test_items_update_reflects_in_get(items_api: APIRequestContext):
    """Value updated via PUT is immediately returned by subsequent GET."""
    item = _create_item(items_api, "Mutable Item", 10.0)
    item_id = item["id"]

    items_api.put(f"/items/{item_id}", data={"name": "Updated Mutable", "price": 20.0})
    fetched = items_api.get(f"/items/{item_id}").json()
    assert fetched["name"] == "Updated Mutable"
    assert fetched["price"] == 20.0


# ===========================================================================
# Extended edge cases
# ===========================================================================

@pytest.mark.items
def test_items_create_zero_price_is_valid(items_api: APIRequestContext):
    """Zero price is accepted (no lower-bound constraint on price in schema)."""
    response = items_api.post("/items", data={"name": "Free Item", "price": 0.0})
    assert response.status == 201


@pytest.mark.items
def test_items_get_with_string_id_returns_422_or_404(items_api: APIRequestContext):
    """GET /items/<string> returns 422 (type error) or 404 (not found)."""
    response = items_api.get("/items/not_an_int", fail_on_status_code=False)
    assert response.status in (404, 422), f"Unexpected status: {response.status}"
