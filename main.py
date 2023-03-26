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

    def __init__(self, *args, **kwargs):
        self._router = APIRouter(*args, **kwargs)

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

    def build(self, obj: object(), ) -> Resource:
        methods = inspect.getmembers(
            obj, predicate=inspect.isfunction
        )

        for method in methods:
            http_methods = self.http_methods(method[0])

            if not http_methods or method[0].startswith('__'):
                continue

            resource_name = multiple_replace(method[0], dict([(i, '') for i in self.__http_methods]))
            resource_name = resource_name[:3].replace('_', '') + resource_name[3:]
            response_model = obj().__getattribute__(f'{resource_name}_response_model') \
                if hasattr(obj, f'{resource_name}_response_model') else None
            include_in_schema = True
            if method[0].startswith('_'):
                include_in_schema = False

            placeholder = self._placeholder(method[1])
            path = '/' + resource_name
            path += placeholder[1]
            endpoint = method[1]
            endpoint.__annotations__ = placeholder[0]
            description, response_description = method[1].__doc__.split('<>')[:2]
            self._router.add_api_route(
                path=path,
                endpoint=method[1],
                methods=http_methods,
                include_in_schema=include_in_schema,
                description=description,
                response_description=response_description,
                response_model=response_model,
            )
        return Resource(router=self._router, scope=obj())


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


foo = MetaRouteClass(tags=['foo'])


@foo.build
class Hidden:
    """
    info
    """
    any_data = 0
    data_response_model = Default

    class ResponseClass:
        def __init__(self, foo: str):
            self.foo = foo

    async def some_method(self):
        return f'{self.ResponseClass.__dict__} {self.any_data}'

    @staticmethod
    async def _post_data(param: Default):
        """default foo method<>default foo answer"""
        return param
    @staticmethod
    async def __some_function():
        return 'go to /data_foo'


foo = MetaRouteClass(tags=['foo'])


@foo.build
class Api:
    param_1: str = None
    param_2: str = None

    @staticmethod
    async def post_data(param1: PlaceholderField(int, 'another_1'), param2: PlaceholderField(int, 'another')):
        """docstring<>docstring"""
        return


# this is a cycle Example
# Info.router.include_router(Info.router)


app = FastAPI(docs_url='/')
Api.router.include_router(Hidden.router)
app.include_router(Api.router)

if __name__ == '__main__':
    asyncio.run(Server(Config(app)).serve())
