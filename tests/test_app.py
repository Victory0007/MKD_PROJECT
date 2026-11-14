from io import BytesIO
from unittest.mock import patch

import app as application


def test_health_check():
    response = application.app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_text_search_returns_products():
    expected = [{
        "id": "product-1",
        "score": 0.91,
        "description": "Pink handbag",
        "stock_code": "BAG-1",
        "unit_price": 12.5,
    }]
    with patch.object(application, "similarity_search", return_value=expected) as search:
        response = application.app.test_client().post(
            "/api/search",
            json={"query": "pink bags"},
        )

    assert response.status_code == 200
    assert response.get_json()["products"] == expected
    search.assert_called_once_with("pink bags", top_k=5)


def test_text_search_rejects_non_string_query():
    with patch.object(
        application,
        "similarity_search",
        side_effect=ValueError("Search query must be a string."),
    ):
        response = application.app.test_client().post(
            "/api/search",
            json={"query": 123},
        )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_ocr_search_requires_file():
    response = application.app.test_client().post("/api/ocr-search")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_ocr_search_rejects_unsupported_extension():
    response = application.app.test_client().post(
        "/api/ocr-search",
        data={"file": (BytesIO(b"not-an-image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "error" in response.get_json()
