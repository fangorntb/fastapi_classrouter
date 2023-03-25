import inspect
import re
from types import UnionType
from typing import Optional, get_origin, Union, Type, List

from fastapi.routing import APIRouter
from pydantic import BaseModel, create_model


def multiple_replace(string: str, rep_dict: dict[str, str]) -> str:
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)


class PlaceholderField:
    def __init__(self, tpe: type):
        if tpe not in (str, int,):
            raise TypeError('tpe for PlaceholderField must be in (str, int, )')
        self.tpe = tpe


class MetaPydanticModel:
    @classmethod
    def dict_model(cls, name: str, dict_def: dict) -> type[BaseModel]:
        fields = {}
        for field_name, value in dict_def.items():
            if isinstance(value, tuple):
                fields[field_name] = value
            elif isinstance(value, dict):
                fields[field_name] = (cls.dict_model(f'{name}_{field_name}', value), ...)
            else:
                raise ValueError(f"Field {field_name}:{value} has invalid syntax")
        return create_model(name, **fields)

    @classmethod
    def _sample_class(cls, obj, ) -> dict:
        annotations = []
        for name, annotation in obj.__annotations__.items():
            default = obj().__getattribute__(name)
            if get_origin(annotation) not in {list, tuple, str, int, bool, None, UnionType, Union, Optional} \
                    or name.startswith("_"):
                continue
            else:
                annotations.append((name, (annotation, default)))
        return dict(annotations)

    @classmethod
    def gen_model(cls, obj: object, ) -> type[BaseModel]:
        return cls.dict_model(obj.__name__, cls._sample_class(obj))


class Resource:
    def __init__(self, scope, router: APIRouter, response: Type[BaseModel], ):
        self.router = router
        self.response = response
        self.scope = scope


class MetaRouteClass(MetaPydanticModel):
    _routes = []
    _router = APIRouter()
    __http_methods = ['get', 'post', 'put', 'purge', 'delete']

    def __init__(self, *args, **kwargs):
        self._router = APIRouter(*args, **kwargs)

    @staticmethod
    def _placeholder(method) -> tuple[dict, str]:
        annotations = dict(method.__annotations__)
        placeholders = ''
        for name, annotation in annotations.items():
            if isinstance(annotation, PlaceholderField):
                annotations[name] = annotation.tpe
                placeholders += "/{" + f'{name}' + "}"
        return annotations, placeholders

    @classmethod
    def http_methods(cls, function_name: str) -> List[str]:
        return list(filter(lambda x: x in function_name, cls.__http_methods))

    def build(self, obj: object()) -> Resource:
        global response_model
        methods = inspect.getmembers(
            obj, predicate=inspect.isfunction
        )
        for method in methods:
            http_methods = self.http_methods(method[0])

            if not http_methods or method[0].startswith('__'):
                continue

            include_in_schema = True
            if method[0].startswith('_'):
                include_in_schema = False
            placeholder = self._placeholder(method[1])
            path = '/' + multiple_replace(method[0], dict([(i, '') for i in self.__http_methods])).lstrip('_')
            path += placeholder[1]
            endpoint = method[1]
            endpoint.__annotations__ = placeholder[0]
            response_model = self.gen_model(obj)
            description, response_description = method[1].__doc__.split('<>')[:2]
            self._router.add_api_route(
                path=path,
                endpoint=method[1],
                methods=http_methods,
                include_in_schema=include_in_schema,
                response_model=response_model,
                description=description,
                response_description=response_description
            )
        return Resource(router=self._router, response=response_model, scope=obj())


# ----------------------------------------------------------------------------------------------------------------------
# Example:

import asyncio

from time import time

from fastapi import FastAPI
from uvicorn import Config, Server


class Default(BaseModel):
    param1: str
    param2: str
    param3: str
    param4: str
    param5: str


foo = MetaRouteClass(tags=['foo'])


@foo.build
class Info:
    param_1: str | Default = None
    param_2: float = ...
    """
    info
    """
    any_data = 0

    class ResponseClass:
        def __init__(self, foo: str):
            self.foo = foo

    def some_method(self):
        return f'{self.ResponseClass.__dict__} {self.any_data}'

    @staticmethod
    def _get_data_foo():
        "default foo method<>default foo answer"
        return Info.response(param_1=Info.scope.some_method(), param_2=time()).dict()

    @staticmethod
    def post_data_foo(data: PlaceholderField(str), param: Default):
        """default foo method<>default foo answer"""
        return Info.response(param_1=Info.scope.__some_function(), param_2=time()).dict()

    @staticmethod
    def __some_function():
        return 'go to /data_foo'

# this is a cycle Example
# Info.router.include_router(Info.router)


app = FastAPI(docs_url='/')
app.include_router(Info.router)

if __name__ == '__main__':
    asyncio.run(Server(Config(app)).serve())
