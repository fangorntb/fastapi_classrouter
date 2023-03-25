import asyncio
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


class PydanticModelBuilder:
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
    def __init__(self, router: APIRouter, response_model: Type[BaseModel]):
        self.router = router
        self.response_model = response_model


class RouteClassFactory(PydanticModelBuilder):
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

    def build(self, obj: object) -> Resource:
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
            path = multiple_replace(f'/{method[0]}', dict([(i, '') for i in self.__http_methods + ['_']]))
            path += placeholder[1]
            endpoint = method[1]
            endpoint.__annotations__ = placeholder[0]
            self._router.prefix += f'/{obj.__name__}'.lower()
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
        return Resource(router=self._router, response_model=response_model)


# ----------------------------------------------------------------------------------------------------------------------
# Example:


from fastapi import FastAPI
from uvicorn import Config, Server


class Default(BaseModel):
    param1: str
    param2: str
    param3: str
    param4: str
    param5: str


foo = RouteClassFactory(tags=['foo'])


@foo.build
class Foo:
    param: str | Default = None
    """
    info
    """

    @staticmethod
    def post_data(data: PlaceholderField(str), param: Default):
        "default foo method<>default foo answer"
        return Foo.response_model(param=param).dict()


app = FastAPI(docs_url='/')
app.include_router(Foo.router)

if __name__ == '__main__':
    asyncio.run(Server(Config(app)).serve())
