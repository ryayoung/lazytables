from __future__ import annotations
from dataclasses import dataclass

from typing import *

from typing import Protocol

TableType = TypeVar("TableType")


def lazytables(
    table_type: Type[TableType],
) -> Callable[[Type], _TableProtocol[TableType]]:
    """
    Lazytables make it easy to manage a large number of data sources in your code.

    This decorator factory makes your class act like a database, where your
    instance is the namespace, and each attribute is a table.

    With a SQL-like syntax for accessing data, you can freely access data sources
    at your leisure, with peace of mind that the data will only be read on-demand
    as you need it, and the same data will never be read twice.

    Lazytables puts all the power and control in your hands. 
    It has no authority over how data is read or written.
    In fact, it doesn't even know how your data is read or written.

    How to use
    ----------
    Have a function of your own that takes a string as input, and returns some data.
    Typically, the string would be the base name of a file, without
    its path or extension. For instance, if you have `/my_data`, a folder with
    a bunch of csv files, you might create a function like this:

        def read_table(name: str) -> pd.DataFrame:
            return pd.read_csv(f'/my_data/{name}.csv')
       
    Optionally, you may choose to define a write function as well. The requirements
    for this function are the same, but it should take an additional argument: the
    data to write:

        def write_table(name: str, data: pd.DataFrame) -> None:
            ...

    Create a `lazytables` class that defines the names of data sources as attributes.
    One argument is required: the type of data that will be returned by your read function.
    This is used for type checking.
        
        @lazytables(pd.DataFrame)
        class MyData:
            ...

    Define attributes to represent the names that will be passed to the read/write functions.
    For each attribute, you have two options:
    1. Define a custom string to be passed to the read/write functions.
       `table_one = 'table-one.xlsx'`
    2. Hint it as an ellipses, to let the name itself be used as the function argument.
       `table_one: ...`

    Create an instance by passing your read, and optionally a write function, to the constructor.
    This instance will work like a namespace for your tables.

        data = MyData(read_table)  # write is optional

    Let's put it all together:

        @lazytables(dict)
        class S3Data:
            table_one: ...
            table_two = 'table-two.csv'
            table_three: ...

        s3 = S3Data(lambda name: pd.read_json(f's3://bucket/{name}'))

        df1 = s3.table_one  # Reads s3://bucket/table_one
        df1 = s3.table_one  # Returns cached result

        df2 = s3.table_two  # Uses '/table-two.csv', NOT '/table_two'

    Writing Data
    ------------
    Each instance you create will have a `write` attribute, which is a namespace for,
    again, all the tables. But this time the table names are functions, which will
    call your write function, and then return the original tables namespace again
    so you can chain writers.

        data.write.table_one(new_df) \
            .write.table_two(new_df2)

    Or, in case you want to write multiple tables and want to pass names dynamically,
    the `write` attribute is also a function, which works the same, but takes the
    table name as an argument, or accepts a dictionary of table names to dataframes.

        data.write('table_one', new_df) \
            .write({'table_two': new_df2})

    Type Checking
    ------------
    This decorator is fully type checked. Your checker will know that each attribute
    is the type you specified. It will enforce that your read function be hinted to
    return that type. And within the `write` namespace, regardless of how you've
    defined your write function, it will know that the second parameter must be
    that that type as well.
    """
    def decorator(cls):
        # Tables
        table_mapping = _get_cls_table_mapping(cls)
        for k, v in table_mapping.items():
            setattr(cls, k, _TableReader(k, v))

        # (Only type hinting this so __annotations__ will be accurate)
        def __init__(
            self,
            reader: Callable[[str], TableType],
            writer: Optional[Callable[[str, TableType, *Tuple[Any, ...]], Any]] = None,
            cache: bool = True,
        ):
            self.write = _TableWriter(self)
            self._read_func = reader
            self._write_func = writer
            self._cache = cache
            self._data = {}

        import re
        # Set an accurate `__annotations__` so that the actual class name of
        # 'table_type' is used instead of 'TableType'
        type_hints = get_type_hints(__init__, globals(), locals())
        hint_name = lambda x: re.sub(r"\w+\.", "", str(x))
        __init__.__annotations__ = {k: hint_name(v) for k, v in type_hints.items()}

        # Extras
        table_type_name = table_type.__name__
        annotations = {k: table_type_name for k in table_mapping.keys()}

        cls._table_mapping = table_mapping
        cls._write_func = None
        cls.__init__ = __init__
        cls.__annotations__ = annotations
        return cls

    return decorator


class _TableProtocol(Protocol[TableType]):
    """
    Inform the type checker of the following:
        - The type of each user-defined attribute (table) in a lazytables
        - The type of each attribute (table writer) under the `write` namespace.
        - The return type and parameter types of the user's read function,
          and write function
        - The return type and parameter types of the user's write function
    """
    _read_func: Callable[[str], TableType]
    _write_func: Callable[[str, TableType, *Tuple[Any, ...]], Any]
    _table_mapping: Dict[str, str]
    _cache: bool
    _data: dict[str, TableType]

    @property
    def write(self) -> _TableWriter[TableType]:
        ...

    def __getattr__(self, table_name: str) -> TableType:
        ...

    def __init__(
        self,
        reader: Callable[[str], TableType],
        writer: Optional[Callable[[str, TableType, *Tuple[Any, ...]], Any]] = None,
        cache: bool = False,
    ):
        ...

    def __call__(
        self,
        reader: Callable[[str], TableType],
        writer: Optional[Callable[[str, TableType, *Tuple[Any, ...]], Any]] = None,
        cache: bool = False,
    ) -> _TableProtocol[TableType]:
        ...


@dataclass(slots=True, frozen=True)
class _TableReader:
    """
    A descriptor who, when accessed, calls the user's read function
    to retrieve a table, or return a cached table if read already.
    """
    key: str
    value: Any

    def __get__(self, inst, _):
        if not inst._cache:
            return inst._read_func(self.value)
        data = inst._data.get(self.key)
        if not data:
            data = inst._read_func(self.value)
        return data


@dataclass(slots=True, frozen=True)
class _TableWriter(Generic[TableType]):
    """
    A namespace (and optionally, callable) owned by a lazytables object that allows
    for writing tables using the user's provided write function.
    Holds a reference to the original lazytables object.

    Key features:
        1. Updates the cache with the written data.
        2. Any table name can be accessed as an attribute, but instead of returning
           a table, it's a callable for the user's write function
    """
    tables: _TableProtocol[TableType]

    def __getattr__(
        self, table_name: str
    ) -> Callable[[TableType, *Tuple[Any, ...]], _TableProtocol[TableType]]:
        """
        Returns a wrapper for `self.__call__` that passes the table name
        as the first argument automatically.

        So instead of ...
        >>> tables.write("table_one", new_df_one)

        ... you can do ...
        >>> tables.write.table_one(new_df_one)
        """
        def write_wrapper(data: TableType, *args, **kwargs) -> _TableProtocol[TableType]:
            return self(table_name, data, *args, **kwargs)
        return write_wrapper

    @overload
    def __call__(self, name_or_mapping: str, data: TableType, **kwargs) -> _TableProtocol[TableType]:
        ...

    @overload
    def __call__(self, name_or_mapping: Dict[str, TableType], data: None, **kwargs) -> _TableProtocol[TableType]:
        ...

    def __call__(self, name_or_mapping, data=None, **kwargs):
        """
        A wrapper for the user's provided write function that:
            1. Executes the user's write function with the provided arguments.
            2. Updates the cache with the new data.
            3. Returns the tables object to allow for method chaining.

        So for a lazytables object, `tables`, you could do:
            (
                tables.write("table_one", new_df_one)
                .write("table_two", new_df_two)
                .write("table_three", new_df_three)
            )
        """
        if not callable(self.tables._write_func):
            raise AttributeError("Lazytables object does not have a write function")

        if (
            (not isinstance(name_or_mapping, dict) and data is None)
            or (isinstance(name_or_mapping, dict) and data is not None)
        ):
            raise ValueError("Must provide either a name and data, or a mapping of names to data")

        mapping = name_or_mapping
        if not isinstance(name_or_mapping, dict):
            data = cast(TableType, data)  # we know it's not None
            mapping = {name_or_mapping: data}

        for table_name, data in mapping.items():
            if table_name not in self.tables._table_mapping:
                raise AttributeError(f"Invalid table name, '{table_name}'.")

            table_id = self.tables._table_mapping[table_name]
            self.tables._write_func(table_id, data, **kwargs)  # Execute the user's write function
            self.tables._data[table_name] = data  # Update the cache
        return self.tables


def _get_cls_table_mapping(cls) -> Dict[str, str]:
    """
    Given a class with attributes annotated or defined with table names
    and/or accessor names, like:

        table_one: ...
        table_two: ... = "table - two (copy)"

    returns a dictionary of attribute names and table names, like:

        {
            'table_one': 'table_one',
            'table_two': 'table - two (copy)'
        }

    Attribute names are used as keys and values by default, but use actual
    values if defined.
    """

    def valid_item(key, value) -> bool:
        if callable(value) or key.startswith("_"):
            return False
        if key in ["read", "write"]:
            raise ValueError(
                f"Invalid table name, '{key}'. If defined, '{key}' must be a function."
            )
        return True

    names_and_values_given = {
        k: v if v is not ... else k for k,v in cls.__dict__.items()
        if valid_item(k, v)
    }
    only_names_given = {
        k: k for k in getattr(cls, "__annotations__", {})
        if valid_item(k, k) and k not in names_and_values_given
    }
    names_and_values_given.update(only_names_given)
    return names_and_values_given





if __name__ == '__main__':
    """
    TESTS:
        Table values:
            key: ...
            key = 'table_name'
            key: ... = 'table_name'

        Input to read func must be string
        Type hint of read func must match table type
        Type hint of write func param must match table type
        Attribute access from class must return table type
    """
    @lazytables(dict)
    class NGAP:
        table_one: dict = {}
        table_two: ...
        table_three: ... = "table - three"
        table_four: ... = ...
        table_five = ...


    def my_read_func_func(name: str) -> dict:
        if name == {}:
            print("Got dict successfully")
        if name is ...:
            print("Didn't get anything")
        return {'name': name, 'a': 1, 'b': 2}

    # def my_write_func(name: str, df: dict) -> None:
        # print("writing", df)

    ngap = NGAP(my_read_func_func)

    # x = ngap.write('table_one', {'a': 1, 'b': 2})
    ngap.write

    for t in [
        ngap.table_one,
        ngap.table_two,
        ngap.table_three,
        ngap.table_four,
        ngap.table_five,
    ]:
        t.update({'c': 1})
        print(t)
        print()

    print(ngap._data)
