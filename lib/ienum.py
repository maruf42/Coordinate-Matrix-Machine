from itertools import islice
from enum import Enum, EnumMeta


class IndexableEnumMeta(EnumMeta):
    """Metaclass for IndexableEnum which can also be accessed like a list through indices."""
    def __getitem__(cls, index):
        if isinstance(index, slice):
            return [cls._member_map_[i] for i in islice(cls._member_map_, index.start, index.stop, index.step)]

        if isinstance(index, int):
            return cls._member_map_[next(islice(cls._member_map_, index, index + 1))]

        return cls._member_map_[index]

    def __repr__(self):
        """A string representation of the key-value pair as a dictionary."""
        return str(self.to_dict())

    def __str__(self):
        """A string representation of the key-value pair as a dictionary."""
        return str(self.to_dict())

    def values(self):
        """Get the list of values."""
        return list(map(lambda c: c.value, self))

    def keys(self):
        """Get the list of names/keys."""
        return list(map(lambda c: c.name, self))

    def to_dict(self):
        """Convert the enum to a dictionary as key-value pairs."""
        return dict(map(lambda c: (c.name, c.value), self))
    
    def to_list(self):
        """Convert the enum to a list as key-value pairs."""
        return list(map(lambda c: (c.name, c.value), self))


class IndexableEnum(Enum, metaclass=IndexableEnumMeta):
    """Creates an Enum which can also be accessed like a list through indices.
    
    Examples:
        >>> from mlaas_py_utilities.config_parser import config_to_enum
        >>> parameters = config_to_enum(open('../test_data/parameters.conf').read(), 'Parameters')

        >>> parameters.keys()
        ['zone', 'pre_process']
        
        >>> parameters.values()
        ['1', 'default']
        
        >>> parameters.to_dict()
        {'zone': '1', 'pre_process': 'default'}
        
        >>> parameters.to_list()
        [('zone', '1'), ('pre_process', 'default')]
    """
    pass
