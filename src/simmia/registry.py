class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        def _register_decorator(func_or_class):
            key = name if name is not None else func_or_class.__name__

            if key in self._module_dict:
                raise KeyError(
                    f"'{key}' is already registered in {self._name} registry"
                )

            self._module_dict[key] = func_or_class

            return func_or_class

        return _register_decorator

    def __getitem__(self, name):
        if name not in self._module_dict:
            raise KeyError(
                f"'{name}' is not found in {self._name} registry. "
                f"Available: {list(self._module_dict.keys())}"
            )
        return self._module_dict[name]

    def list_keys(self):
        return list(self._module_dict.keys())

    def __repr__(self):
        return f"Registry(name={self._name}, items={self._module_dict})"

    def __contains__(self, key):
        return key in self._module_dict
