from rag_med.qa_generator import generate_qa


def test_generate_qa(mocker) -> None:
    """Тест генерации вопросов-ответов из текстового фрагмента."""
    chunk = (
        "Лекарства, повышающие артериальное давление, могут использоваться для лечения гипертонии."
    )

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "text": "Вопрос: Что такое гипертония?\nОтвет: Устойчивое повышение артериального давления.",
                    }
                ]
            }

    mocker.patch("rag_med.qa_generator.generator.requests.post", return_value=_Resp())

    result = generate_qa(chunk, chunk_index=1)

    assert result.chunk_index == 1
    assert result.chunk == chunk
    assert len(result.chunk) > 0
    assert result.model_used is not None
    assert result.question != ""
    assert result.answer != ""
