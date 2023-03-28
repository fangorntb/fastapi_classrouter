import inspect
import re
from typing import List

from fastapi.routing import APIRouter
from pydantic import BaseModel


def multiple_replace(string: str, rep_dict: dict[str, str]) -> str:
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict, key=len, reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)


class PlaceholderField:
    def __init__(self, tpe: type, res: str = None):
        if tpe not in (str, int,):
            raise TypeError('tpe for PlaceholderField must be in (str, int, )')
        self.tpe = tpe
        self.res = res


class Resource:
    def __init__(self, scope, router: APIRouter, ):
        self.router = router
        self.scope = scope


class MetaRouteClass:
    _routes = []
    _router = APIRouter()

    __http_methods = ['get', 'post', 'put', 'patch', 'delete']

    @staticmethod
    def _placeholder(method) -> tuple[dict, str]:
        annotations = dict(method.__annotations__)
        placeholders = ''
        for name, annotation in annotations.items():
            if isinstance(annotation, PlaceholderField):
                annotations[name] = annotation.tpe
                placeholders += "/{" + f'{name}' + "}" +\
                    f"/{annotation.res}" if annotation.res is not None else ''
        return annotations, placeholders

    @classmethod
    def http_methods(cls, function_name: str) -> List[str]:
        return list(filter(lambda x: x in function_name, cls.__http_methods))

    @classmethod
    def build(cls, obj: object(), ) -> Resource:
        methods = inspect.getmembers(
            obj, predicate=inspect.isfunction
        )

        for method in methods:
            http_methods = cls.http_methods(method[0])

            if not http_methods or method[0].startswith('__'):
                continue

            resource_name = multiple_replace(method[0], dict([(i, '') for i in cls.__http_methods]))
            resource_name = resource_name[:3].replace('_', '') + resource_name[3:]
            include_in_schema = True
            if method[0].startswith('_'):
                include_in_schema = False

            placeholder = cls._placeholder(method[1])
            path = '/' + resource_name
            path += placeholder[1]
            endpoint = method[1]
            endpoint.__annotations__ = placeholder[0]
            response_model = endpoint.__annotations__.get('return')
            description, response_description = method[1].__doc__.split('<>')[:2]
            cls._router.add_api_route(
                path=path,
                name=resource_name,
                tags=[obj.__name__],
                endpoint=method[1],
                methods=http_methods,
                include_in_schema=include_in_schema,
                description=description,
                response_description=response_description,
                response_model=response_model,
            )
        return Resource(router=cls._router, scope=obj())


# ----------------------------------------------------------------------------------------------------------------------
# Example:

import asyncio

from fastapi import FastAPI
from uvicorn import Config, Server


class Default(BaseModel):
    param1: str
    param2: str
    param3: str
    param4: str
    param5: str


foo = MetaRouteClass()


@foo.build
class Hidden:
    """
    info
    """
    any_data = 0

    class ResponseClass:
        def __init__(self, foo: str):
            self.foo = foo

    async def some_method(self):
        return f'{self.ResponseClass.__dict__} {self.any_data}'

    @staticmethod
    async def _post_data(param: Default) -> Default:
        """default foo method<>default foo answer"""
        return param
    @staticmethod
    async def __some_function():
        return 'go to /data_foo'
