"""Small test modules, each a way an environment may reach what pytest prints.

``tests/test_environment_assertions.py`` runs each program under the static
check and under pytest, and the two must agree but for the lists it holds. So
these programs are what the check promises: a new way to leak goes here as a
flow, a source or a sink, and either the check flags it or the test's list of
known misses names it.

A program combines:

- a flow, ``(name, definitions, setup, expression)``: module-level code, the
  first lines of the test, and an expression holding the value. ``{S}`` in any
  of them stands for the source. A setup of ``PARAM name`` makes ``name`` a
  parameter of the test, and ``MARK name`` parametrizes the test as ``MARKS``
  says, its parameter holding the value;
- a source, an expression: an environment, or something safe to print;
- a sink, ``(definitions, body)``: how the test fails with the value, ``{V}`` in
  the body standing for the flow's expression.

Trying every combination would run some 150,000 programs, so ``programs``
crosses each flow with every sink on one source, and with every source on one
sink.
"""

from __future__ import annotations

from typing import NamedTuple

SOURCES = {
    "dict": "dict(os.environ)",
    "environ": "os.environ",
    "copy": "os.environ.copy()",
    "star": "{**os.environ}",
    "comp": "{k: v for k, v in os.environ.items()}",
    "items": "list(os.environ.items())",
    "values": "list(os.environ.values())",
    "join": '" ".join(os.environ.values())',
    "str": "str(os.environ)",
    "fstr": 'f"{os.environ}"',
    "dumps": "json.dumps(dict(os.environ))",
    "deepcopy": "copy.deepcopy(dict(os.environ))",
}

D = "dataclasses.dataclass"

FLOWS = [
    # ---- direct and local bindings
    ("direct", "", "", "{S}"),
    ("assign", "", "value = {S}", "value"),
    ("assign_chain", "", "a = value = {S}", "a"),
    ("annotated", "", "value: dict = {S}", "value"),
    ("walrus_if", "", "if (value := {S}):\n    pass", "value"),
    ("walrus_inline", "", "", "(value := {S})"),
    ("walrus_comp", "", "[(value := {S}) for _ in range(1)]", "value"),
    ("unpack_pair", "", "_, value = 1, {S}", "value"),
    ("unpack_list", "", "_, value = [1, {S}]", "value"),
    ("unpack_nested", "", "(_, (value, _)) = (1, ({S}, 2))", "value"),
    ("unpack_star", "", "first, *value = 1, {S}", "value"),
    ("unpack_star_last", "", "*_, value = [1, {S}]", "value"),
    ("aug_or", "", 'value = {}\nvalue |= {"k": {S}}', "value"),
    ("aug_add", "", "value = [1]\nvalue += [{S}]", "value"),
    ("for_target", "", "for value in [{S}]:\n    pass", "value"),
    ("for_tuple", "", "for _, value in [(1, {S})]:\n    pass", "value"),
    ("for_dict_items", "", 'for _, value in {"k": {S}}.items():\n    pass', "value"),
    ("for_values", "", 'for value in {"k": {S}}.values():\n    pass', "value"),
    ("for_enumerate", "", "for _, value in enumerate([{S}]):\n    pass", "value"),
    ("for_zip", "", "for _, value in zip([1], [{S}]):\n    pass", "value"),
    ("while_assign", "", "while True:\n    value = {S}\n    break", "value"),
    ("if_assign", "", "if True:\n    value = {S}", "value"),
    ("try_assign", "", "try:\n    value = {S}\nfinally:\n    pass", "value"),
    ("with_null", "", "with contextlib.nullcontext():\n    value = {S}", "value"),
    (
        "with_as_null",
        "",
        "with contextlib.nullcontext({S}) as value:\n    pass",
        "value",
    ),
    ("list_comp", "", "value = [x for x in [{S}]][0]", "value"),
    ("set_item", "", 'value = {}\nvalue["k"] = {S}', 'value["k"]'),
    ("set_item_whole", "", 'value = {}\nvalue["k"] = {S}', "value"),
    ("dict_literal", "", 'value = {"k": {S}}', 'value["k"]'),
    ("dict_literal_whole", "", 'value = {"k": {S}}', "value"),
    ("list_literal", "", "value = [{S}]", "value[0]"),
    ("list_append", "", "value = []\nvalue.append({S})", "value"),
    ("list_extend", "", "value = []\nvalue.extend([{S}])", "value"),
    ("list_insert", "", "value = []\nvalue.insert(0, {S})", "value"),
    ("deque_append", "", "value = collections.deque()\nvalue.append({S})", "value"),
    ("set_add", "", "value = set()\nvalue.add(str({S}))", "value"),
    ("dict_update", "", "value = {}\nvalue.update(k={S})", "value"),
    ("dict_setdefault", "", 'value = {}\nvalue.setdefault("k", {S})', "value"),
    ("dict_get", "", 'mapping = {"k": {S}}', 'mapping.get("k")'),
    ("dict_pop", "", 'mapping = {"k": {S}}', 'mapping.pop("k")'),
    ("slice", "", "items = [{S}]", "items[:1]"),
    ("ifexp", "", "", "({S} if True else None)"),
    ("boolop_or", "", "", "(None or {S})"),
    ("boolop_and", "", "", "(1 and {S})"),
    ("tuple_wrap", "", "", "({S},)"),
    ("set_wrap", "", "", "{str({S})}"),
    ("list_call", "", "", "list([{S}])"),
    ("sorted_call", "", "", "sorted([str({S})])"),
    ("tuple_call", "", "", "tuple([{S}])"),
    ("dict_kw", "", "", "dict(k={S})"),
    ("copy_copy", "", "", "copy.copy({S})"),
    ("str_call", "", "", "str({S})"),
    ("repr_call", "", "", "repr({S})"),
    ("format_call", "", "", '"{}".format({S})'),
    ("percent", "", "", '"%s" % ({S},)'),
    ("fstring", "", "", 'f"{{S}}"'),
    # ---- match
    ("match_capture", "", "match {S}:\n    case value:\n        pass", "value"),
    (
        "match_as",
        "",
        "match {S}:\n    case dict() | str() | list() | os._Environ() as value:\n        pass",
        "value",
    ),
    ("match_seq", "", "match [{S}]:\n    case [value]:\n        pass", "value"),
    ("match_star", "", "match [1, {S}]:\n    case [_, *value]:\n        pass", "value"),
    (
        "match_map",
        "",
        'match {"k": {S}}:\n    case {"k": value}:\n        pass',
        "value",
    ),
    ("match_rest", "", 'match {"k": {S}}:\n    case {**value}:\n        pass', "value"),
    (
        "match_class_kw",
        "@" + D + "\nclass Holder:\n    data: object",
        "match Holder({S}):\n    case Holder(data=value):\n        pass",
        "value",
    ),
    (
        "match_class_pos",
        "@" + D + "\nclass Holder:\n    data: object",
        "match Holder({S}):\n    case Holder(value):\n        pass",
        "value",
    ),
    ("match_guard", "", "match {S}:\n    case value if value:\n        pass", "value"),
    # ---- exceptions
    (
        "except_as",
        "",
        "try:\n    raise ValueError({S})\nexcept ValueError as caught:\n    value = caught",
        "value",
    ),
    (
        "except_args",
        "",
        "try:\n    raise ValueError({S})\nexcept ValueError as caught:\n    value = caught.args",
        "value",
    ),
    # ---- defaults and closures
    ("lambda_default", "", "f = lambda x={S}: x", "f()"),
    ("def_default", "def f(x={S}):\n    return x", "", "f()"),
    ("kw_default", "def f(*, x={S}):\n    return x", "", "f()"),
    ("closure_local", "", "captured = {S}\ndef f():\n    return captured", "f()"),
    # ---- globals
    ("global_read", "G = {S}", "", "G"),
    ("global_ann", "G: dict = {S}", "", "G"),
    ("global_alias", "G = {S}\nH = G", "", "H"),
    ("global_unpack", "_, G = 1, {S}", "", "G"),
    ("global_if", "if True:\n    G = {S}", "", "G"),
    ("global_for", "for G in [{S}]:\n    pass", "", "G"),
    ("global_with", "with contextlib.nullcontext({S}) as G:\n    pass", "", "G"),
    (
        "global_try",
        "try:\n    raise ValueError({S})\nexcept ValueError as caught:\n    G = caught",
        "",
        "G",
    ),
    ("global_walrus", "(G := {S})", "", "G"),
    ("global_match", "match {S}:\n    case G:\n        pass", "", "G"),
    ("global_func", "G = None\ndef fill():\n    global G\n    G = {S}", "fill()", "G"),
    ("global_func_nodecl", "def fill():\n    global G\n    G = {S}", "fill()", "G"),
    ("global_store", 'G = {}\ndef fill():\n    G["k"] = {S}', "fill()", 'G["k"]'),
    ("global_store_whole", 'G = {}\ndef fill():\n    G["k"] = {S}', "fill()", "G"),
    ("global_append", "G = []\ndef fill():\n    G.append({S})", "fill()", "G"),
    (
        "global_in_nested",
        "G = None",
        "def fill():\n    global G\n    G = {S}\nfill()",
        "G",
    ),
    ("global_comp", "G = [x for x in [{S}]]", "", "G"),
    ("global_import", "from os import environ as G", "", "G"),
    (
        "nonlocal_nested",
        "",
        "value = None\ndef fill():\n    nonlocal value\n    value = {S}\nfill()",
        "value",
    ),
    # ---- function returns
    ("fn", "def f():\n    return {S}", "", "f()"),
    ("fn_assign", "def f():\n    return {S}", "value = f()", "value"),
    ("fn_if", "if True:\n    def f():\n        return {S}", "", "f()"),
    ("fn_for", "for _ in [1]:\n    def f():\n        return {S}", "", "f()"),
    ("fn_try", "try:\n    def f():\n        return {S}\nfinally:\n    pass", "", "f()"),
    (
        "fn_with",
        "with contextlib.nullcontext():\n    def f():\n        return {S}",
        "",
        "f()",
    ),
    ("fn_alias_module", "def f():\n    return {S}\ng = f", "", "g()"),
    ("fn_alias_chain", "def f():\n    return {S}\ng = f\nh = g", "", "h()"),
    ("fn_alias_local", "def f():\n    return {S}", "g = f", "g()"),
    ("fn_alias_walrus", "def f():\n    return {S}", "", "(g := f)()"),
    ("fn_alias_unpack", "def f():\n    return {S}\n_, g = 1, f", "", "g()"),
    ("fn_ifexp", "def f():\n    return {S}", "", "(f if True else None)()"),
    ("fn_boolop", "def f():\n    return {S}", "", "(None or f)()"),
    (
        "fn_param",
        "def f():\n    return {S}\ndef call(g):\n    return g()",
        "",
        "call(f)",
    ),
    (
        "fn_param_kw",
        "def f():\n    return {S}\ndef call(*, g):\n    return g()",
        "",
        "call(g=f)",
    ),
    ("fn_dict", "def f():\n    return {S}", "", '{"k": f}["k"]()'),
    ("fn_list", "def f():\n    return {S}", "", "[f][0]()"),
    (
        "fn_global_dict",
        'def f():\n    return {S}\nTABLE = {"k": f}',
        "",
        'TABLE["k"]()',
    ),
    ("fn_partial", "def f():\n    return {S}", "", "functools.partial(f)()"),
    ("fn_partial_arg", "def f(x):\n    return x", "", "functools.partial(f, {S})()"),
    ("fn_decorated_cache", "@functools.lru_cache\ndef f():\n    return {S}", "", "f()"),
    (
        "fn_decorated_identity",
        "def deco(g):\n    return g\n@deco\ndef f():\n    return {S}",
        "",
        "f()",
    ),
    (
        "fn_decorated_wrapper",
        "def deco(g):\n    def wrapper(*args):\n        return g(*args)\n    return wrapper\n@deco\ndef f():\n    return {S}",
        "",
        "f()",
    ),
    (
        "fn_factory",
        "def make():\n    def f():\n        return {S}\n    return f",
        "",
        "make()()",
    ),
    (
        "fn_factory_alias",
        "def make():\n    def f():\n        return {S}\n    return f",
        "g = make()",
        "g()",
    ),
    ("fn_factory_lambda", "def make():\n    return lambda: {S}", "", "make()()"),
    ("fn_through", "def f():\n    return {S}\ndef g():\n    return f()", "", "g()"),
    (
        "fn_through_assign",
        "def f():\n    return {S}\ndef g():\n    value = f()\n    return value",
        "",
        "g()",
    ),
    ("fn_tuple_return", "def f():\n    return 1, {S}", "_, value = f()", "value"),
    ("fn_tuple_index", "def f():\n    return 1, {S}", "", "f()[1]"),
    ("fn_dict_return", 'def f():\n    return {"k": {S}}', "", 'f()["k"]'),
    (
        "fn_conditional_return",
        "def f():\n    if True:\n        return {S}\n    return None",
        "",
        "f()",
    ),
    (
        "fn_try_return",
        "def f():\n    try:\n        return {S}\n    finally:\n        pass",
        "",
        "f()",
    ),
    ("fn_identity", "def f(x):\n    return x", "", "f({S})"),
    ("fn_identity_kw", "def f(x):\n    return x", "", "f(x={S})"),
    ("fn_star", "def f(*args):\n    return args[0]", "", "f({S})"),
    ("fn_starstar", 'def f(**kwargs):\n    return kwargs["x"]', "", "f(x={S})"),
    ("fn_star_call", "def f(x):\n    return x", "args = [{S}]", "f(*args)"),
    ("fn_starstar_call", "def f(x):\n    return x", 'args = {"x": {S}}', "f(**args)"),
    (
        "fn_fill_param",
        'def fill(box):\n    box["k"] = {S}',
        "value = {}\nfill(value)",
        "value",
    ),
    (
        "fn_fill_list",
        "def fill(box):\n    box.append({S})",
        "value = []\nfill(value)",
        "value",
    ),
    ("lambda_module", "f = lambda: {S}", "", "f()"),
    ("lambda_alias", "f = lambda: {S}\ng = f", "", "g()"),
    ("lambda_local", "", "f = lambda: {S}", "f()"),
    ("lambda_local_alias", "", "f = lambda: {S}\ng = f", "g()"),
    ("lambda_now", "", "", "(lambda: {S})()"),
    ("lambda_param", "def call(g):\n    return g()", "", "call(lambda: {S})"),
    ("lambda_arg", "f = lambda x: x", "", "f({S})"),
    ("lambda_dict", 'TABLE = {"k": lambda: {S}}', "", 'TABLE["k"]()'),
    ("lambda_list", "", "", "[lambda: {S}][0]()"),
    ("nested_def", "", "def f():\n    return {S}", "f()"),
    ("nested_alias", "", "def f():\n    return {S}\ng = f", "g()"),
    ("nested_param", "", "def f(x):\n    return x", "f({S})"),
    ("nested_if", "", "if True:\n    def f():\n        return {S}", "f()"),
    ("async_run", "async def f():\n    return {S}", "", "asyncio.run(f())"),
    (
        "async_await",
        "async def f():\n    return {S}\nasync def g():\n    return await f()",
        "",
        "asyncio.run(g())",
    ),
    # ---- generators
    ("gen_next", "def g():\n    yield {S}", "", "next(g())"),
    ("gen_list", "def g():\n    yield {S}", "", "list(g())"),
    ("gen_for", "def g():\n    yield {S}", "for value in g():\n    pass", "value"),
    ("gen_comp", "def g():\n    yield {S}", "", "[x for x in g()]"),
    (
        "gen_from",
        "def g():\n    yield {S}\ndef h():\n    yield from g()",
        "",
        "next(h())",
    ),
    ("gen_from_list", "def h():\n    yield from [{S}]", "", "list(h())"),
    ("gen_alias", "def g():\n    yield {S}\nh = g", "", "next(h())"),
    (
        "gen_tuple",
        "def g():\n    yield 1, {S}",
        "for _, value in g():\n    pass",
        "value",
    ),
    ("gen_exp_next", "", "", "next(x for x in [{S}])"),
    ("gen_exp_list", "", "", "list(x for x in [{S}])"),
    (
        "ctx_manager",
        "@contextlib.contextmanager\ndef c():\n    yield {S}",
        "with c() as value:\n    pass",
        "value",
    ),
    (
        "ctx_alias",
        "@contextlib.contextmanager\ndef c():\n    yield {S}\nd = c",
        "with d() as value:\n    pass",
        "value",
    ),
    ("map_call", "def f(x):\n    return x", "", "list(map(f, [{S}]))"),
    ("filter_call", "", "", "list(filter(None, [{S}]))"),
    # ---- class bodies
    ("cls_attr", "class C:\n    x = {S}", "", "C.x"),
    ("cls_attr_inst", "class C:\n    x = {S}", "", "C().x"),
    ("cls_attr_ann", "class C:\n    x: dict = {S}", "", "C.x"),
    ("cls_attr_if", "class C:\n    if True:\n        x = {S}", "", "C.x"),
    ("cls_attr_for", "class C:\n    for x in [{S}]:\n        pass", "", "C.x"),
    (
        "cls_attr_while",
        "class C:\n    while True:\n        x = {S}\n        break",
        "",
        "C.x",
    ),
    (
        "cls_attr_with",
        "class C:\n    with contextlib.nullcontext({S}) as x:\n        pass",
        "",
        "C.x",
    ),
    (
        "cls_attr_try",
        "class C:\n    try:\n        x = {S}\n    finally:\n        pass",
        "",
        "C.x",
    ),
    (
        "cls_attr_except",
        "class C:\n    try:\n        raise ValueError({S})\n    except ValueError as caught:\n        x = caught",
        "",
        "C.x",
    ),
    ("cls_attr_unpack", "class C:\n    _, x = 1, {S}", "", "C.x"),
    ("cls_attr_walrus", "class C:\n    (x := {S})", "", "C.x"),
    (
        "cls_attr_match",
        "class C:\n    match {S}:\n        case x:\n            pass",
        "",
        "C.x",
    ),
    ("cls_attr_import", "class C:\n    from os import environ as x", "", "C.x"),
    (
        "cls_attr_import_mod",
        "class C:\n    import os as system\n    x = system.environ",
        "",
        "C.x",
    ),
    ("cls_attr_alias", "class C:\n    y = {S}\n    x = y", "", "C.x"),
    ("cls_attr_comp", "class C:\n    x = [i for i in [{S}]]", "", "C.x"),
    ("cls_attr_global", "G = {S}\nclass C:\n    x = G", "", "C.x"),
    ("cls_attr_fn", "def f():\n    return {S}\nclass C:\n    x = f()", "", "C.x"),
    ("cls_attr_store", "class C:\n    pass\nC.x = {S}", "", "C.x"),
    ("cls_attr_store_inst", "class C:\n    pass\nC.x = {S}", "", "C().x"),
    ("cls_attr_store_test", "class C:\n    pass", "C.x = {S}", "C.x"),
    ("cls_attr_setattr", 'class C:\n    pass\nsetattr(C, "x", {S})', "", "C.x"),
    ("cls_attr_getattr", "class C:\n    x = {S}", "", 'getattr(C, "x")'),
    ("cls_attr_vars", "class C:\n    x = {S}", "", 'vars(C)["x"]'),
    ("cls_attr_dict", "class C:\n    x = {S}", "", 'C.__dict__["x"]'),
    ("cls_attr_alias_cls", "class C:\n    x = {S}\nD = C", "", "D.x"),
    ("cls_attr_inherited", "class B:\n    x = {S}\nclass C(B):\n    pass", "", "C.x"),
    (
        "cls_attr_inherited_inst",
        "class B:\n    x = {S}\nclass C(B):\n    pass",
        "",
        "C().x",
    ),
    ("cls_attr_if_class", "if True:\n    class C:\n        x = {S}", "", "C.x"),
    ("cls_attr_nested", "class Outer:\n    class C:\n        x = {S}", "", "Outer.C.x"),
    ("cls_attr_local", "", "class C:\n    x = {S}", "C.x"),
    ("cls_attr_local_inst", "", "class C:\n    x = {S}", "C().x"),
    # ---- methods
    ("m", "class C:\n    def f(self):\n        return {S}", "", "C().f()"),
    ("m_assign", "class C:\n    def f(self):\n        return {S}", "c = C()", "c.f()"),
    (
        "m_static",
        "class C:\n    @staticmethod\n    def f():\n        return {S}",
        "",
        "C.f()",
    ),
    (
        "m_static_inst",
        "class C:\n    @staticmethod\n    def f():\n        return {S}",
        "",
        "C().f()",
    ),
    (
        "m_class",
        "class C:\n    @classmethod\n    def f(cls):\n        return {S}",
        "",
        "C.f()",
    ),
    ("m_unbound", "class C:\n    def f(self):\n        return {S}", "", "C.f(None)"),
    (
        "m_bound_alias",
        "class C:\n    def f(self):\n        return {S}",
        "g = C().f",
        "g()",
    ),
    (
        "m_bound_alias_module",
        "class C:\n    def f(self):\n        return {S}\ng = C().f",
        "",
        "g()",
    ),
    (
        "m_unbound_alias",
        "class C:\n    def f(self):\n        return {S}\ng = C.f",
        "",
        "g(None)",
    ),
    (
        "m_getattr",
        "class C:\n    def f(self):\n        return {S}",
        "",
        'getattr(C(), "f")()',
    ),
    (
        "m_if",
        "class C:\n    if True:\n        def f(self):\n            return {S}",
        "",
        "C().f()",
    ),
    (
        "m_for",
        "class C:\n    for _ in [1]:\n        def f(self):\n            return {S}",
        "",
        "C().f()",
    ),
    (
        "m_try",
        "class C:\n    try:\n        def f(self):\n            return {S}\n    finally:\n        pass",
        "",
        "C().f()",
    ),
    (
        "m_inherited",
        "class B:\n    def f(self):\n        return {S}\nclass C(B):\n    pass",
        "",
        "C().f()",
    ),
    (
        "m_if_class",
        "if True:\n    class C:\n        def f(self):\n            return {S}",
        "",
        "C().f()",
    ),
    (
        "m_nested_class",
        "class Outer:\n    class C:\n        def f(self):\n            return {S}",
        "",
        "Outer.C().f()",
    ),
    ("m_local_class", "", "class C:\n    def f(self):\n        return {S}", "C().f()"),
    ("m_lambda_attr", "class C:\n    f = lambda self: {S}", "", "C().f()"),
    (
        "m_assigned_fn",
        "def g(self):\n    return {S}\nclass C:\n    f = g",
        "",
        "C().f()",
    ),
    (
        "m_assigned_later",
        "def g(self):\n    return {S}\nclass C:\n    pass\nC.f = g",
        "",
        "C().f()",
    ),
    ("m_inst_lambda", "class C:\n    pass", "c = C()\nc.f = lambda: {S}", "c.f()"),
    (
        "m_inst_fn_attr",
        "def g():\n    return {S}\nclass C:\n    pass",
        "c = C()\nc.f = g",
        "c.f()",
    ),
    (
        "m_self_attr",
        "class C:\n    def __init__(self):\n        self.x = {S}\n    def f(self):\n        return self.x",
        "",
        "C().f()",
    ),
    (
        "m_self_call",
        "class C:\n    def g(self):\n        return {S}\n    def f(self):\n        return self.g()",
        "",
        "C().f()",
    ),
    ("m_param", "class C:\n    def f(self, x):\n        return x", "", "C().f({S})"),
    (
        "m_super",
        "class B:\n    def f(self):\n        return {S}\nclass C(B):\n    def f(self):\n        return super().f()",
        "",
        "C().f()",
    ),
    (
        "m_property",
        "class C:\n    @property\n    def p(self):\n        return {S}",
        "",
        "C().p",
    ),
    (
        "m_cached_property",
        "class C:\n    @functools.cached_property\n    def p(self):\n        return {S}",
        "",
        "C().p",
    ),
    (
        "m_property_if",
        "class C:\n    if True:\n        @property\n        def p(self):\n            return {S}",
        "",
        "C().p",
    ),
    (
        "m_property_call",
        "class C:\n    def _p(self):\n        return {S}\n    p = property(_p)",
        "",
        "C().p",
    ),
    # ---- special methods
    ("d_call", "class C:\n    def __call__(self):\n        return {S}", "", "C()()"),
    (
        "d_call_alias",
        "class C:\n    def __call__(self):\n        return {S}",
        "c = C()",
        "c()",
    ),
    (
        "d_getitem",
        "class C:\n    def __getitem__(self, key):\n        return {S}",
        "",
        'C()["k"]',
    ),
    (
        "d_getitem_if",
        "class C:\n    if True:\n        def __getitem__(self, key):\n            return {S}",
        "",
        'C()["k"]',
    ),
    (
        "d_getattr",
        "class C:\n    def __getattr__(self, name):\n        return {S}",
        "",
        "C().x",
    ),
    (
        "d_getattribute",
        "class C:\n    def __getattribute__(self, name):\n        return {S}",
        "",
        "C().x",
    ),
    (
        "d_enter",
        "class C:\n    def __enter__(self):\n        return {S}\n    def __exit__(self, *args):\n        return False",
        "with C() as value:\n    pass",
        "value",
    ),
    (
        "d_iter",
        "class C:\n    def __iter__(self):\n        yield {S}",
        "for value in C():\n    pass",
        "value",
    ),
    (
        "d_iter_list",
        "class C:\n    def __iter__(self):\n        return iter([{S}])",
        "",
        "list(C())",
    ),
    (
        "d_next",
        "class C:\n    def __iter__(self):\n        return self\n    def __next__(self):\n        return {S}",
        "",
        "next(C())",
    ),
    (
        "d_repr_attr",
        "class C:\n    def __init__(self):\n        self.x = {S}\n    def __repr__(self):\n        return repr(self.x)",
        "",
        "C()",
    ),
    (
        "d_repr_direct",
        "class C:\n    def __repr__(self):\n        return repr({S})",
        "",
        "C()",
    ),
    (
        "d_str_exc",
        "class E(Exception):\n    def __str__(self):\n        return str({S})",
        "",
        "E()",
    ),
    (
        "d_add",
        "class C:\n    def __add__(self, other):\n        return {S}",
        "",
        "C() + 1",
    ),
    (
        "d_missing",
        "class C(dict):\n    def __missing__(self, key):\n        return {S}",
        "",
        'C()["k"]',
    ),
    (
        "d_get",
        "class D:\n    def __get__(self, obj, owner=None):\n        return {S}\nclass C:\n    x = D()",
        "",
        "C().x",
    ),
    (
        "d_class_getitem",
        "class C:\n    def __class_getitem__(cls, key):\n        return {S}",
        "",
        'C["k"]',
    ),
    # ---- instances
    (
        "i_init_attr",
        "class C:\n    def __init__(self):\n        self.x = {S}",
        "",
        "C().x",
    ),
    (
        "i_init_param",
        "class C:\n    def __init__(self, x):\n        self.x = x",
        "",
        "C({S}).x",
    ),
    (
        "i_init_kw",
        "class C:\n    def __init__(self, x):\n        self.x = x",
        "",
        "C(x={S}).x",
    ),
    (
        "i_init_if",
        "class C:\n    if True:\n        def __init__(self):\n            self.x = {S}",
        "",
        "C().x",
    ),
    (
        "i_method_store",
        "class C:\n    def fill(self):\n        self.x = {S}",
        "c = C()\nc.fill()",
        "c.x",
    ),
    ("i_attr_store", "class C:\n    pass", "c = C()\nc.x = {S}", "c.x"),
    ("i_attr_store_module", "class C:\n    pass\nc = C()\nc.x = {S}", "", "c.x"),
    ("i_setattr", "class C:\n    pass", 'c = C()\nsetattr(c, "x", {S})', "c.x"),
    ("i_getattr", "class C:\n    pass", "c = C()\nc.x = {S}", 'getattr(c, "x")'),
    (
        "i_getattr_var",
        "class C:\n    pass",
        'c = C()\nc.x = {S}\nname = "x"',
        "getattr(c, name)",
    ),
    ("i_vars", "class C:\n    pass", "c = C()\nc.x = {S}", "vars(c)"),
    ("i_vars_item", "class C:\n    pass", "c = C()\nc.x = {S}", 'vars(c)["x"]'),
    ("i_dunder_dict", "class C:\n    pass", "c = C()\nc.x = {S}", "c.__dict__"),
    ("i_dict_store", "class C:\n    pass", 'c = C()\nc.__dict__["x"] = {S}', "c.x"),
    ("i_alias", "class C:\n    pass", "c = C()\nc.x = {S}\nd = c", "d.x"),
    ("i_list", "class C:\n    pass", "c = C()\nc.x = {S}\nitems = [c]", "items[0].x"),
    (
        "i_param",
        "class C:\n    pass\ndef read(obj):\n    return obj.x",
        "c = C()\nc.x = {S}",
        "read(c)",
    ),
    (
        "i_param_method",
        "class C:\n    def read(self):\n        return self.x",
        "c = C()\nc.x = {S}",
        "c.read()",
    ),
    (
        "i_inherit_init",
        "class B:\n    def __init__(self):\n        self.x = {S}\nclass C(B):\n    pass",
        "",
        "C().x",
    ),
    (
        "i_inherit_super",
        "class B:\n    def __init__(self, x):\n        self.x = x\nclass C(B):\n    def __init__(self):\n        super().__init__({S})",
        "",
        "C().x",
    ),
    (
        "i_slots",
        'class C:\n    __slots__ = ("x",)\n    def __init__(self):\n        self.x = {S}',
        "",
        "C().x",
    ),
    (
        "i_repr_vars",
        "class C:\n    def __init__(self):\n        self.x = {S}\n    def __repr__(self):\n        return repr(vars(self))",
        "",
        "C()",
    ),
    ("i_namespace", "", "value = types.SimpleNamespace(x={S})", "value"),
    ("i_namespace_attr", "", "value = types.SimpleNamespace()\nvalue.x = {S}", "value"),
    ("i_namespace_read", "", "value = types.SimpleNamespace(x={S})", "value.x"),
    ("i_mock_attr", "", "value = unittest.mock.Mock()\nvalue.x = {S}", "value.x"),
    ("i_mock_call", "", "value = unittest.mock.Mock()\nvalue({S})", "value.call_args"),
    ("i_mock_return", "", "value = unittest.mock.Mock(return_value={S})", "value()"),
    # ---- exceptions as values
    ("e_args", "class E(Exception):\n    pass", "", "E({S})"),
    ("e_args_read", "class E(Exception):\n    pass", "", "E({S}).args"),
    (
        "e_self_args",
        "class E(Exception):\n    def fill(self):\n        self.args = ({S},)\n        return self",
        "",
        "E().fill()",
    ),
    (
        "e_init_args",
        "class E(Exception):\n    def __init__(self):\n        self.args = ({S},)",
        "",
        "E()",
    ),
    (
        "e_init_super",
        "class E(Exception):\n    def __init__(self):\n        super().__init__({S})",
        "",
        "E()",
    ),
    (
        "e_oserror_filename",
        'class E(OSError):\n    def __init__(self):\n        self.errno = 1\n        self.strerror = "s"\n        self.filename = {S}',
        "",
        "str(E())",
    ),
    (
        "e_note",
        "class E(Exception):\n    def fill(self):\n        self.add_note(str({S}))\n        return self",
        "",
        "E().fill().__notes__",
    ),
    # ---- records
    ("r_kw", "@" + D + "\nclass R:\n    data: object", "", "R(data={S})"),
    ("r_pos", "@" + D + "\nclass R:\n    data: object", "", "R({S})"),
    ("r_env", "@" + D + "\nclass R:\n    env: object", "", "R(env={S})"),
    ("r_read", "@" + D + "\nclass R:\n    data: object", "", "R(data={S}).data"),
    (
        "r_assign",
        "@" + D + "\nclass R:\n    data: object",
        "value = R(data={S})",
        "value",
    ),
    (
        "r_starstar",
        "@" + D + "\nclass R:\n    data: object",
        'args = {"data": {S}}',
        "R(**args)",
    ),
    ("r_star", "@" + D + "\nclass R:\n    data: object", "args = [{S}]", "R(*args)"),
    ("r_default", "@" + D + "\nclass R:\n    data: object = ({S},)", "", "R()"),
    (
        "r_default_read",
        "@" + D + "\nclass R:\n    data: object = ({S},)",
        "",
        "R().data",
    ),
    (
        "r_field_default",
        "@" + D + "\nclass R:\n    data: object = dataclasses.field(default=({S},))",
        "",
        "R()",
    ),
    (
        "r_factory_lambda",
        "@"
        + D
        + "\nclass R:\n    data: object = dataclasses.field(default_factory=lambda: {S})",
        "",
        "R()",
    ),
    (
        "r_factory_fn",
        "def make():\n    return {S}\n@"
        + D
        + "\nclass R:\n    data: object = dataclasses.field(default_factory=make)",
        "",
        "R()",
    ),
    (
        "r_if",
        "@" + D + "\nclass R:\n    if True:\n        data: object",
        "",
        "R(data={S})",
    ),
    ("r_alias", "@" + D + "\nclass R:\n    data: object\nA = R", "", "A(data={S})"),
    ("r_alias_local", "@" + D + "\nclass R:\n    data: object", "A = R", "A(data={S})"),
    (
        "r_param_class",
        "@"
        + D
        + "\nclass R:\n    data: object\ndef build(kind):\n    return kind(data={S})",
        "",
        "build(R)",
    ),
    (
        "r_dict_class",
        "@" + D + "\nclass R:\n    data: object",
        "",
        '{"k": R}["k"](data={S})',
    ),
    (
        "r_partial",
        "@" + D + "\nclass R:\n    data: object",
        "",
        "functools.partial(R, data={S})()",
    ),
    (
        "r_type",
        "@" + D + "\nclass R:\n    data: object",
        "seed = R(data=1)",
        "type(seed)(data={S})",
    ),
    (
        "r_class_attr",
        "@" + D + "\nclass R:\n    data: object",
        "seed = R(data=1)",
        "seed.__class__(data={S})",
    ),
    (
        "r_classmethod",
        "@"
        + D
        + "\nclass R:\n    data: object\n    @classmethod\n    def make(cls):\n        return cls(data={S})",
        "",
        "R.make()",
    ),
    (
        "r_staticmethod",
        "@"
        + D
        + "\nclass R:\n    data: object\n    @staticmethod\n    def make():\n        return R(data={S})",
        "",
        "R.make()",
    ),
    (
        "r_inherit",
        "@"
        + D
        + "\nclass B:\n    data: object\n@"
        + D
        + "\nclass R(B):\n    other: object = 1",
        "",
        "R(data={S})",
    ),
    (
        "r_post_init",
        "@"
        + D
        + "\nclass R:\n    data: object = None\n    def __post_init__(self):\n        self.data = {S}",
        "",
        "R()",
    ),
    (
        "r_method_store",
        "@"
        + D
        + "\nclass R:\n    data: object = None\n    def fill(self):\n        self.data = {S}\n        return self",
        "",
        "R().fill()",
    ),
    (
        "r_attr_store",
        "@" + D + "\nclass R:\n    data: object = None",
        "value = R()\nvalue.data = {S}",
        "value",
    ),
    (
        "r_asdict",
        "@" + D + "\nclass R:\n    data: object",
        "",
        "dataclasses.asdict(R(data={S}))",
    ),
    (
        "r_astuple",
        "@" + D + "\nclass R:\n    data: object",
        "",
        "dataclasses.astuple(R(data={S}))",
    ),
    (
        "r_replace",
        "@" + D + "\nclass R:\n    data: object",
        "",
        "dataclasses.replace(R(data=1), data={S})",
    ),
    (
        "r_replace_star",
        "@" + D + "\nclass R:\n    data: object",
        "",
        'dataclasses.replace(R(data=1), **{"data": {S}})',
    ),
    (
        "r_replace_keep",
        "@" + D + "\nclass R:\n    data: object\n    other: object = 1",
        "",
        "dataclasses.replace(R(data={S}), other=2)",
    ),
    ("r_vars", "@" + D + "\nclass R:\n    data: object", "", "vars(R(data={S}))"),
    (
        "r_getattr",
        "@" + D + "\nclass R:\n    data: object",
        "",
        'getattr(R(data={S}), "data")',
    ),
    (
        "r_frozen",
        "@dataclasses.dataclass(frozen=True)\nclass R:\n    data: object",
        "",
        "R(data={S})",
    ),
    (
        "r_slots",
        "@dataclasses.dataclass(slots=True)\nclass R:\n    data: object",
        "",
        "R(data={S})",
    ),
    (
        "r_kw_only",
        "@dataclasses.dataclass(kw_only=True)\nclass R:\n    data: object",
        "",
        "R(data={S})",
    ),
    (
        "r_fn_param_annot",
        "@"
        + D
        + "\nclass R:\n    env: object\ndef read(record: R):\n    return record",
        "",
        "read(R(env={S}))",
    ),
    ("r_local_class", "", "@" + D + "\nclass R:\n    data: object", "R(data={S})"),
    (
        "r_nested_class",
        "class Outer:\n    @" + D + "\n    class R:\n        data: object",
        "",
        "Outer.R(data={S})",
    ),
    (
        "r_if_class",
        "if True:\n    @" + D + "\n    class R:\n        data: object",
        "",
        "R(data={S})",
    ),
    ("r_list", "@" + D + "\nclass R:\n    data: object", "", "[R(data={S})]"),
    ("r_dict", "@" + D + "\nclass R:\n    data: object", "", '{"k": R(data={S})}'),
    # record through an aliased dataclasses module, or imported names
    (
        "ra_dc_field",
        "import dataclasses as dc\n@dc.dataclass\nclass R:\n    data: object = dc.field(default_factory=lambda: {S})",
        "",
        "R()",
    ),
    (
        "ra_dc_field_default",
        "import dataclasses as dc\n@dc.dataclass\nclass R:\n    data: object = dc.field(default=({S},))",
        "",
        "R()",
    ),
    (
        "ra_dc_replace",
        "import dataclasses as dc\n@dc.dataclass\nclass R:\n    data: object",
        "",
        "dc.replace(R(data=1), data={S})",
    ),
    (
        "ra_dc_replace_star",
        "import dataclasses as dc\n@dc.dataclass\nclass R:\n    data: object",
        "",
        'dc.replace(R(data=1), **{"data": {S}})',
    ),
    (
        "ra_dc_asdict",
        "import dataclasses as dc\n@dc.dataclass\nclass R:\n    data: object",
        "",
        "dc.asdict(R(data={S}))",
    ),
    (
        "ra_from_field",
        "from dataclasses import dataclass as dcl, field as fl\n@dcl\nclass R:\n    data: object = fl(default_factory=lambda: {S})",
        "",
        "R()",
    ),
    (
        "ra_from_replace",
        "from dataclasses import dataclass as dcl, replace as rp\n@dcl\nclass R:\n    data: object",
        "",
        "rp(R(data=1), data={S})",
    ),
    (
        "ra_from_asdict",
        "from dataclasses import dataclass as dcl, asdict as ad\n@dcl\nclass R:\n    data: object",
        "",
        "ad(R(data={S}))",
    ),
    (
        "ra_local_import",
        "",
        'import dataclasses as dc\nrecord = dc.make_dataclass("R", ["data"])({S})',
        "record",
    ),
    (
        "ra_make_dataclass",
        'R = dataclasses.make_dataclass("R", ["data"])',
        "",
        "R({S})",
    ),
    # named tuples and others
    ("n_kw", "class N(typing.NamedTuple):\n    data: object", "", "N(data={S})"),
    ("n_pos", "class N(typing.NamedTuple):\n    data: object", "", "N({S})"),
    ("n_default", "class N(typing.NamedTuple):\n    data: object = {S}", "", "N()"),
    ("n_read", "class N(typing.NamedTuple):\n    data: object", "", "N({S}).data"),
    ("n_index", "class N(typing.NamedTuple):\n    data: object", "", "N({S})[0]"),
    (
        "n_unpack",
        "class N(typing.NamedTuple):\n    data: object",
        "value, = N({S})",
        "value",
    ),
    (
        "n_alias_module",
        "import typing as t\nclass N(t.NamedTuple):\n    data: object",
        "",
        "N({S})",
    ),
    (
        "n_from_alias",
        "from typing import NamedTuple as NT\nclass N(NT):\n    data: object",
        "",
        "N({S})",
    ),
    (
        "n_replace",
        "class N(typing.NamedTuple):\n    data: object",
        "",
        "N(1)._replace(data={S})",
    ),
    (
        "n_asdict",
        "class N(typing.NamedTuple):\n    data: object",
        "",
        "N({S})._asdict()",
    ),
    ("n_collections", 'N = collections.namedtuple("N", "data")', "", "N({S})"),
    ("td", "class T(typing.TypedDict):\n    data: object", "", "T(data={S})"),
    ("enum_member", "class K(enum.Enum):\n    A = {S}", "", "K.A"),
    ("enum_value", "class K(enum.Enum):\n    A = {S}", "", "K.A.value"),
    (
        "metaclass_repr",
        "class M(type):\n    def __repr__(cls):\n        return repr(cls.x)\nclass C(metaclass=M):\n    x = {S}",
        "",
        "C",
    ),
    (
        "class_decorator",
        "def deco(cls):\n    cls.__repr__ = lambda self: repr(self.x)\n    return cls\n@deco\nclass C:\n    def __init__(self):\n        self.x = {S}",
        "",
        "C()",
    ),
    # ---- fixtures and parametrize
    ("fx", "@pytest.fixture\ndef fx():\n    return {S}", "PARAM fx", "fx"),
    ("fx_yield", "@pytest.fixture\ndef fx():\n    yield {S}", "PARAM fx", "fx"),
    (
        "fx_if",
        "if True:\n    @pytest.fixture\n    def fx():\n        return {S}",
        "PARAM fx",
        "fx",
    ),
    (
        "fx_chain",
        "@pytest.fixture\ndef base():\n    return {S}\n@pytest.fixture\ndef fx(base):\n    return base",
        "PARAM fx",
        "fx",
    ),
    (
        "fx_autouse_global",
        'G = {}\n@pytest.fixture(autouse=True)\ndef fill():\n    G["k"] = {S}',
        "",
        "G",
    ),
    (
        "fx_name",
        '@pytest.fixture(name="fx")\ndef _make():\n    return {S}',
        "PARAM fx",
        "fx",
    ),
    (
        "fx_factory",
        "@pytest.fixture\ndef fx():\n    def make():\n        return {S}\n    return make",
        "PARAM fx",
        "fx()",
    ),
    (
        "fx_request",
        "@pytest.fixture\ndef fx():\n    return {S}",
        "PARAM request",
        'request.getfixturevalue("fx")',
    ),
    (
        "fx_param",
        "@pytest.fixture(params=[{S}])\ndef fx(request):\n    return request.param",
        "PARAM fx",
        "fx",
    ),
    ("param_mark", "", "MARK parametrize", "value"),
    ("param_pytest_param", "", "MARK pytest_param", "value"),
    ("param_list", "VALUES = [{S}]", "MARK values", "value"),
    ("param_multi", "", "MARK multi", "value"),
    # ---- other module forms
    ("import_system", "import os as system\nG = system.environ", "", "G"),
    ("from_environ", "from os import environ as E", "", "E"),
    ("from_environ_local", "", "from os import environ as E", "E"),
    ("import_local", "", "import os as system", "system.environ"),
    ("sys_modules", "import sys", "", 'sys.modules["os"].environ'),
    ("importlib_os", "import importlib", "", 'importlib.import_module("os").environ'),
    ("dunder_import", "", "", '__import__("os").environ'),
    ("getattr_os", "", "", 'getattr(os, "environ")'),
    ("subprocess_os", "import subprocess", "", "subprocess.os.environ"),
    ("os_path_os", "", "", "os.path.os.environ"),
    ("global_ns", "", 'globals()["G"] = {S}', 'globals()["G"]'),
    ("locals_ns", "", "value = {S}", "locals()"),
    ("exec_eval", "", "", "eval('''{S}''')"),
    # ---- text methods
    ("text_strip", "", "", "str({S}).strip()"),
    ("text_lower", "", "value = str({S}).lower()", "value"),
    ("text_split", "", "", "str({S}).split()"),
    ("text_encode", "", "", "str({S}).encode()"),
    ("text_replace", "", "", 'str({S}).replace("not-there", "")'),
    ("text_join_str", "", "", '"".join(str({S}))'),
    ("text_format_map", "", "", '"{x}".format_map({"x": {S}})'),
    # ---- local classes that leave their function
    (
        "m_local_returned",
        "def make():\n    class C:\n        def f(self):\n            return {S}\n    return C()",
        "",
        "make().f()",
    ),
    (
        "m_local_class_returned",
        "def make():\n    class C:\n        def f(self):\n            return {S}\n    return C",
        "",
        "make()().f()",
    ),
    (
        "cls_local_returned",
        "def make():\n    class C:\n        x = {S}\n    return C",
        "",
        "make().x",
    ),
    (
        "i_local_returned",
        "def make():\n    class C:\n        def __init__(self):\n            self.x = {S}\n    return C()",
        "",
        "make().x",
    ),
    (
        "m_local_inherited",
        "",
        "class B:\n    def f(self):\n        return {S}\nclass C(B):\n    pass",
        "C().f()",
    ),
    ("cls_local_inherited", "", "class B:\n    x = {S}\nclass C(B):\n    pass", "C.x"),
    (
        "i_local_inherited",
        "",
        "class B:\n    def __init__(self):\n        self.x = {S}\nclass C(B):\n    pass",
        "C().x",
    ),
    # ---- what a later definition replaces
    (
        "fn_redefined",
        "def f():\n    return 1\nif True:\n    def f():\n        return {S}",
        "",
        "f()",
    ),
    (
        "cls_redefined",
        "class C:\n    x = 1\nif True:\n    class C:\n        x = {S}",
        "",
        "C.x",
    ),
    (
        "m_redefined",
        "class C:\n    def f(self):\n        return 1\n    if True:\n        def f(self):\n            return {S}",
        "",
        "C().f()",
    ),
    (
        "m_redefined_class",
        "class C:\n    def f(self):\n        return 1\nif True:\n    class C:\n        def f(self):\n            return {S}",
        "",
        "C().f()",
    ),
    # ---- two steps, under other names
    (
        "fn_returns_instance",
        "class C:\n    def f(self):\n        return {S}\ndef make():\n    return C()",
        "",
        "make().f()",
    ),
    (
        "fn_returns_class",
        "class C:\n    x = {S}\ndef kind():\n    return C",
        "",
        "kind().x",
    ),
    (
        "fn_returns_class_built",
        "class C:\n    x = {S}\ndef kind():\n    return C",
        "",
        "kind()().x",
    ),
    (
        "m_returns_itself",
        "class C:\n    def me(self):\n        return self\n    def f(self):\n        return {S}",
        "",
        "C().me().f()",
    ),
    (
        "m_instance_attr",
        "class C:\n    def f(self):\n        return {S}\nclass D:\n    def __init__(self):\n        self.c = C()",
        "",
        "D().c.f()",
    ),
    (
        "m_class_attr_instance",
        "class C:\n    def f(self):\n        return {S}\nclass D:\n    c = C()",
        "",
        "D.c.f()",
    ),
    ("m_in_list", "class C:\n    def f(self):\n        return {S}", "", "[C()][0].f()"),
    (
        "m_in_dict",
        "class C:\n    def f(self):\n        return {S}",
        "",
        '{"k": C()}["k"].f()',
    ),
    (
        "m_on_other_class",
        "class C:\n    def f(self):\n        return {S}\nclass D:\n    g = C.f",
        "",
        "D().g()",
    ),
    (
        "m_getattr_name",
        'class C:\n    def f(self):\n        return {S}\nNAME = "f"',
        "",
        "getattr(C(), NAME)()",
    ),
    (
        "m_generator",
        "class C:\n    def g(self):\n        yield {S}",
        "",
        "next(C().g())",
    ),
    (
        "m_context",
        "class C:\n    @contextlib.contextmanager\n    def c(self):\n        yield {S}",
        "with C().c() as value:\n    pass",
        "value",
    ),
    (
        "fn_partial_alias",
        "def f():\n    return {S}\ng = functools.partial(f)\nh = g",
        "",
        "h()",
    ),
    (
        "fn_static_attr",
        "def f():\n    return {S}\nclass C:\n    g = staticmethod(f)",
        "",
        "C.g()",
    ),
    (
        "fn_static_attr_inst",
        "def f():\n    return {S}\nclass C:\n    g = staticmethod(f)",
        "",
        "C().g()",
    ),
    (
        "r_returned_read",
        "@" + D + "\nclass R:\n    data: object\ndef make():\n    return R(data={S})",
        "",
        "make().data",
    ),
    (
        "r_in_class_attr",
        "@" + D + "\nclass R:\n    data: object\nclass C:\n    r = R(data={S})",
        "",
        "C.r.data",
    ),
    (
        "mock_return_read",
        "",
        "value = unittest.mock.Mock()\nvalue.f.return_value = {S}",
        "value.f()",
    ),
    (
        "mock_call_list",
        "",
        "value = unittest.mock.Mock()\nvalue({S})",
        "value.call_args_list",
    ),
    (
        "mock_call_kwargs",
        "",
        "value = unittest.mock.Mock()\nvalue(x={S})",
        "value.call_args.kwargs",
    ),
    (
        "mock_method_call",
        "",
        "value = unittest.mock.Mock()\nvalue.f({S})",
        "value.f.call_args",
    ),
    # ---- wrappers around definitions
    (
        "deco_method",
        "def deco(g):\n    return g\nclass C:\n    @deco\n    def f(self):\n        return {S}",
        "",
        "C().f()",
    ),
    (
        "deco_wraps",
        "def deco(g):\n    @functools.wraps(g)\n    def wrapper(*args, **kwargs):\n        return g(*args, **kwargs)\n    return wrapper\n@deco\ndef f():\n    return {S}",
        "",
        "f()",
    ),
    (
        "deco_factory",
        "def deco():\n    def inner(g):\n        return g\n    return inner\n@deco()\ndef f():\n    return {S}",
        "",
        "f()",
    ),
    (
        "deco_class",
        "def deco(c):\n    return c\n@deco\nclass C:\n    x = {S}",
        "",
        "C.x",
    ),
    (
        "deco_class_method",
        "def deco(c):\n    return c\n@deco\nclass C:\n    def f(self):\n        return {S}",
        "",
        "C().f()",
    ),
]

SINKS = {
    "is_none": ("", "assert {V} is None"),
    "eq": ("", "assert {V} == 1"),
    "not": ("", "assert not {V}"),
    "len": ("", "assert len({V}) == -1"),
    "call_arg": ("def check(x):\n    return False", "assert check({V})"),
    "raise": ("", "raise ValueError({V})"),
    "raise_named": (
        "class Named(Exception):\n    def __init__(self, detail):\n        super().__init__(detail)",
        "raise Named(detail={V})",
    ),
    "fail": ("", "pytest.fail(str({V}))"),
    "print": ("", "print({V})\nassert False"),
    "log": ("", "logging.error({V})\nassert False"),
    "helper_raise": ('def explode(x):\n    raise ValueError("no")', "explode({V})"),
    "msg": ("", "assert False, {V}"),
    "method_raise": (
        'class Checker:\n    def check(self, x):\n        raise ValueError("no")',
        "Checker().check({V})",
    ),
    "subprocess_env": ("", 'subprocess.run(["/nonexistent-program"], env={V})'),
    "imported_call": ("import textwrap", "textwrap.indent({V}, 1)"),
    "warn": ("", "warnings.warn(str({V}))\nassert False"),
    "attr_read": ("", "assert ({V}).__class__ is None"),
    "subscript_fail": ("", 'assert ({V})["AAA_NOT_THERE_KEY"] is None'),
}

SAFE_SOURCES = {
    "names": "sorted(os.environ)",
    "len": "len(os.environ)",
    "in": '"AAA_OTHER" in os.environ',
    "get": 'os.environ.get("AAA_OTHER")',
    "index": 'os.environ["AAA_OTHER"]',
    "keys": "list(os.environ.keys())",
    "bool": "bool(os.environ)",
    "constant": "1",
}


MARKS = {
    "parametrize": '@pytest.mark.parametrize("value", [{S}])',
    "pytest_param": '@pytest.mark.parametrize("value", [pytest.param({S}, id="x")])',
    "values": '@pytest.mark.parametrize("value", VALUES)',
    "multi": '@pytest.mark.parametrize("a, value", [(1, {S})])',
}
HEADER = """import asyncio, collections, contextlib, copy, dataclasses, enum, functools
import json, logging, os, subprocess, types, typing, unittest.mock, warnings
import pytest

"""


class Program(NamedTuple):
    flow: str
    source: str
    sink: str
    text: str


def _indented(text: str) -> str:
    return "".join(f"    {line}\n" if line else "\n" for line in text.split("\n"))


def _program(
    number: int, flow: tuple[str, str, str, str], source: str, sink: str
) -> str:
    _, definitions, setup, expression = flow
    sink_definitions, sink_body = SINKS[sink]
    parameters = []
    decorator = ""
    if setup.startswith("PARAM "):
        parameters = [setup.split()[1]]
        setup = ""
    elif setup.startswith("MARK "):
        mark = setup.split()[1]
        decorator = MARKS[mark] + "\n"
        parameters = ["a", "value"] if mark == "multi" else ["value"]
        setup = ""
    body = (setup + "\n" if setup else "") + sink_body.replace("{V}", expression)
    text = (
        HEADER
        + (sink_definitions + "\n\n" if sink_definitions else "")
        + (definitions + "\n\n" if definitions else "")
        + decorator
        + f"def test_{number:05d}({', '.join(parameters)}):\n"
        + _indented(body)
    )
    return text.replace("{S}", source)


def programs() -> list[Program]:
    """Every flow with: each sink, given a copy of the environment; each other
    source that holds an environment, then each safe one, failing the same way.
    A program's number is its position in the list."""
    held = "dict"
    combinations = [
        *((flow, held, sink) for flow in FLOWS for sink in SINKS),
        *(
            (flow, source, "is_none")
            for flow in FLOWS
            for source in [*SOURCES, *SAFE_SOURCES]
            if source != held
        ),
    ]
    sources = SOURCES | SAFE_SOURCES
    return [
        Program(flow[0], source, sink, _program(number, flow, sources[source], sink))
        for number, (flow, source, sink) in enumerate(combinations)
    ]
