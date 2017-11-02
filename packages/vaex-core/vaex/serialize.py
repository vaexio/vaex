import vaex.utils
  
registry = {}

def register(cls):
    registry[fullname(cls)] = cls
    return cls

def fullname(cls):
    return cls.__module__ + '.' + cls.__name__

def to_dict(obj):
    return dict(cls=fullname(obj.__class__), state=obj.state_get())

def from_dict(d):
    cls_name = d['cls']
    if cls_name not in registry:
        raise ValueError('unknown class: ' + cls_name)
    else:
        obj = registry[cls_name].state_from(d['state'])
        #obj.state_set(d['state'])
        return obj

def can_serialize(object):
    return fullname(object.__class__) in registry