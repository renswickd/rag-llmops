import pytest
from core.exceptions import RagAssistantException


def test_exception_with_plain_message():
    exc = RagAssistantException("Something went wrong")

    assert exc.error_message == "Something went wrong"
    assert "Something went wrong" in str(exc)
    assert isinstance(exc.lineno, int)
    assert isinstance(exc.file_name, str)


def test_exception_captures_traceback_from_passed_exception():
    try:
        1 / 0
    except ZeroDivisionError as e:
        exc = RagAssistantException("Division failed", e)

    assert exc.error_message == "Division failed"
    assert "Division failed" in str(exc)
    assert "ZeroDivisionError" in exc.traceback_str
    assert "Traceback:" in str(exc)
    assert exc.lineno > 0
    assert exc.file_name.endswith(".py")


def test_exception_uses_sys_exc_info_when_no_error_details_passed():
    try:
        {}["x"]
    except KeyError:
        exc = RagAssistantException("Lookup failed")

    assert exc.error_message == "Lookup failed"
    assert "KeyError" in exc.traceback_str
    assert exc.lineno > 0


def test_repr_output():
    exc = RagAssistantException("Test repr")
    result = repr(exc)

    assert "message='Test repr'" in result
    assert "file=" in result
    assert "line=" in result