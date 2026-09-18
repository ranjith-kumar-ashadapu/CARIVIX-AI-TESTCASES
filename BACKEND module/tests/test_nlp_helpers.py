from nlp_helpers import clean_text, analyze_sentiment, extract_entities, top_keywords


def test_clean_text_lowercases_and_strips_urls():
    result = clean_text("Check this out: https://example.com NOW!")
    assert "https" not in result
    assert result == result.lower()


def test_analyze_sentiment_detects_positive():
    result = analyze_sentiment("Strong growth and record profit this quarter")
    assert result["label"] == "Positive"


def test_analyze_sentiment_detects_negative():
    result = analyze_sentiment("Sharp decline and major losses reported")
    assert result["label"] == "Negative"


def test_analyze_sentiment_returns_expected_keys():
    result = analyze_sentiment("Neutral statement about the weather")
    assert "label" in result
    assert "score" in result


def test_extract_entities_finds_money_and_percent():
    text = "Revenue grew by 12% to ₹500 crore this year"
    entities = extract_entities(text)
    assert "PERCENT" in entities
    assert "MONEY" in entities


def test_top_keywords_returns_list_of_tuples():
    result = top_keywords("data data data pipeline pipeline validation", n=2)
    assert result[0][0] == "data"
    assert result[0][1] == 3