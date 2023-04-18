import inspect
import re
from typing import List

from fastapi.routing import APIRouter


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
    __http_methods = ['get', 'post', 'put', 'patch', 'delete']

    def __init__(self, *args, **kwargs):
        self._route = APIRouter(*args, **kwargs)

    @staticmethod
    def _placeholder(method) -> tuple[dict, str]:
        annotations = dict(method.__annotations__)
        placeholders = ''
        for name, annotation in annotations.items():
            if isinstance(annotation, PlaceholderField):
                annotations[name] = annotation.tpe

                placeholders += str(f"/{annotation.res}" if annotation.res is not None else '') + "/{" + f'{name}' + "}"
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

            include_in_schema = True
            if method[0].startswith('_'):
                include_in_schema = False
            placeholder = self._placeholder(method[1])
            path = placeholder[1]
            path += '/' + resource_name.replace('_', '/') if resource_name != '' else ''

            endpoint = method[1]
            endpoint.__annotations__ = placeholder[0]
            response_model = endpoint.__annotations__.get('return')
            resource_name = re.search(r'<n>(.+?)</n>',  method[1].__doc__).group(1) \
                if re.search(r'<n>(.+?)</n>',  method[1].__doc__) is not None else ''
            description, response_description = method[1].__doc__.replace('\n', '<br>').replace(f'<n>{resource_name}</n>', '').split('<>')[:2]

            self._route.add_api_route(
                path=path,
                tags=[obj.__name__],
                name=resource_name,
                endpoint=method[1],
                methods=http_methods,
                include_in_schema=include_in_schema,
                description=str(description),
                response_description=str(response_description),
                response_model=response_model,
            )
        return Resource(router=self._route, scope=obj())
