from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="Simple Items API (FastAPI)")

# Pydantic models
class ItemBase(BaseModel):
    name: str = Field(..., example="Widget")
    description: Optional[str] = Field(None, example="A useful widget")
    price: float = Field(..., example=9.99)

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

# In-memory "database"
_items: Dict[int, Item] = {}
_next_id = 1

# Helper functions
def _get_next_id() -> int:
    global _next_id
    nid = _next_id
    _next_id += 1
    return nid

# Endpoints (4 endpoints as requested)

@app.get("/items", response_model=List[Item])
def list_items():
    """List all items"""
    return list(_items.values())

@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    """Get a single item by id"""
    item = _items.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/items", response_model=Item, status_code=201)
def create_item(payload: ItemCreate):
    """Create a new item"""
    item_id = _get_next_id()
    item = Item(id=item_id, **payload.dict())
    _items[item_id] = item
    return item

@app.put("/items/{item_id}", response_model=Item)
def update_item(item_id: int, payload: ItemCreate):
    """Update an existing item"""
    existing = _items.get(item_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Item not found")
    updated = Item(id=item_id, **payload.dict())
    _items[item_id] = updated
    return updated

# Optional: populate with example items on startup
@app.on_event("startup")
def startup_populate():
    global _items, _next_id
    if not _items:
        a = Item(id=_get_next_id(), name="Sample A", description="First sample item", price=1.23)
        b = Item(id=_get_next_id(), name="Sample B", description="Second sample item", price=4.56)
        _items[a.id] = a
        _items[b.id] = b

if __name__ == "__main__":
    # Run with: python fastapi_main.py OR use: uvicorn fastapi_main:app --reload
    uvicorn.run("fastapi_main:app", host="127.0.0.1", port=8000, reload=True)
