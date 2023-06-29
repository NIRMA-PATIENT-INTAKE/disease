import pytest


@pytest.fixture(scope="session")
def simple_data():
    texts = ["У меня температура, но нет недомогания", "У меня температура"]
    diseases = ["a", "b"]

    return texts, diseases


@pytest.fixture(scope="session")
def complex_data():
    texts = [
        "У меня температура, но нет недомогания",
        "У меня температура",
        "Температура поднялась",
        "Болит живот",
        "Недомогание не беспокоит, боюсь за изжогу",
        "Изгожа беспокоит, температура нет",
        "Дома холодно, но мне жарко. Кажется у меня температура",
        "Температура меня не беспокоит.",
    ]
    diseases = ["a", "b", "a", "b", "a", "b", "b", "a"]
    return texts, diseases
