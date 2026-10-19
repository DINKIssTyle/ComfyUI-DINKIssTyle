"""Local editor endpoints, independent of the ComfyUI execution queue."""
import asyncio
import io
import json
from urllib.parse import urlsplit
from aiohttp import web
from PIL import Image
from server import PromptServer
from .core.service import service

PREFIX = '/dinki/facemorph'
BUSY = asyncio.Semaphore(1)


def same_origin(request):
    origin = request.headers.get('Origin')
    if origin and urlsplit(origin).netloc != request.host:
        raise web.HTTPForbidden(text='Cross-origin writes are not supported.')


async def body(request):
    same_origin(request)
    if request.content_type != 'application/json':
        raise web.HTTPUnsupportedMediaType()
    raw = bytearray()
    async for chunk in request.content.iter_chunked(65536):
        raw.extend(chunk)
        if len(raw)>2*1024*1024:
            raise web.HTTPRequestEntityTooLarge(max_size=2*1024*1024,actual_size=len(raw))
    data = json.loads(raw)
    if not isinstance(data,dict):
        raise ValueError('Expected an object.')
    return data


def errors(fn):
    async def wrapped(request):
        try:
            return await fn(request)
        except web.HTTPException:
            raise
        except (ValueError,KeyError,TypeError,FileNotFoundError,ImportError,RuntimeError,OSError) as exc:
            return web.json_response({'error':str(exc)},status=400)
    return wrapped


@PromptServer.instance.routes.get(PREFIX+'/models')
@errors
async def models(request):
    return web.json_response(service().status())


@PromptServer.instance.routes.post(PREFIX+'/models')
@errors
async def install(request):
    data = await body(request)
    async with BUSY:
        result = await asyncio.to_thread(service().install,data.get('kind'))
    return web.json_response(result)


@PromptServer.instance.routes.post(PREFIX+'/upload')
@errors
async def upload(request):
    same_origin(request)
    reader = await request.multipart()
    part = await reader.next()
    if part is None or part.name != 'image':
        raise ValueError('Expected an image upload.')
    raw = bytearray()
    while chunk := await part.read_chunk():
        raw.extend(chunk)
        if len(raw)>32*1024*1024:
            raise web.HTTPRequestEntityTooLarge(max_size=32*1024*1024,actual_size=len(raw))
    async with BUSY:
        result = await asyncio.to_thread(service().upload,raw)
    return web.json_response(result)


@PromptServer.instance.routes.post(PREFIX+'/analyze')
@errors
async def analyze(request):
    data = await body(request)
    async with BUSY:
        result = await asyncio.to_thread(service().analyze,data['id'],data.get('face_index',0))
    return web.json_response(result)


@PromptServer.instance.routes.post(PREFIX+'/prepare')
@errors
async def prepare(request):
    data = await body(request)
    async with BUSY:
        result = await asyncio.to_thread(service().prepare,data)
    return web.json_response(result)


@PromptServer.instance.routes.get(PREFIX+'/image/{asset_id}')
@errors
async def image(request):
    path = service().path(request.match_info['asset_id'])
    def preview():
        with Image.open(path) as opened:
            image = opened.convert('RGB')
            if request.query.get('crop'):
                try:bounds=[int(v) for v in request.query['crop'].split(',')]
                except ValueError:raise ValueError('Invalid source crop.')
                if len(bounds)!=4 or not (0<=bounds[0]<bounds[2]<=image.width and 0<=bounds[1]<bounds[3]<=image.height) or min(bounds[2]-bounds[0],bounds[3]-bounds[1])<2:
                    raise ValueError('Invalid source crop.')
                image=image.crop(bounds)
            image.thumbnail((1024,1024))
            buffer = io.BytesIO()
            image.save(buffer,format='JPEG',quality=92)
        return buffer.getvalue()
    result = await asyncio.to_thread(preview)
    return web.Response(body=result,content_type='image/jpeg',headers={'Cache-Control':'private, max-age=86400'})


@PromptServer.instance.routes.get(PREFIX+'/mask/{mask_id}')
@errors
async def mask_image(request):
    from .core.hair import mask_path
    path=mask_path(service(),request.match_info['mask_id'])
    content=await asyncio.to_thread(path.read_bytes)
    return web.Response(body=content,content_type='image/png',headers={'Cache-Control':'private, max-age=31536000, immutable'})


@PromptServer.instance.routes.post(PREFIX+'/mask')
@errors
async def correct_mask(request):
    from .core.geometry import validate_state
    from .core import hair
    data=await body(request)
    asset=data['asset']
    validate_state({'version':1,'assets':[asset,asset],'base':0})
    async with BUSY:
        ids,raw_ids,_=await asyncio.to_thread(hair.prepare,service(),[asset])
    return web.json_response({'mask_ids':ids,'raw_mask_ids':raw_ids})


@PromptServer.instance.routes.post(PREFIX+'/sources')
@errors
async def sources(request):
    data=await body(request)
    async with BUSY:
        result=await asyncio.to_thread(service().sources,data)
    return web.json_response(result)
