import covaudit


def test_package_imports_and_has_version():
    assert isinstance(covaudit.__version__, str)
    assert covaudit.__version__.count(".") == 2
