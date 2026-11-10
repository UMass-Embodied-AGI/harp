"""Lossless primitive packing for the neutral laptop's NYX visual cache.

The legacy per-link GLBs retain dozens of separate nodes. This candidate for
a native NYX scene-import abort packs identical-material triangle batches to
reduce that hierarchy while preserving physics and the exact visual data.
Unsupported glTF features fail explicitly instead of silently being dropped.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


_DTYPES = {5120: '<i1', 5121: '<u1', 5122: '<i2', 5123: '<u2',
           5125: '<u4', 5126: '<f4'}
_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_glb(blob):
    """Read the two standard chunks without re-encoding textures."""
    if len(blob) < 28 or struct.unpack_from('<III', blob) != (0x46546C67, 2, len(blob)):
        raise ValueError('invalid GLB header')
    length, kind = struct.unpack_from('<II', blob, 12)
    if kind != 0x4E4F534A:
        raise ValueError('GLB JSON must be first')
    tree = json.loads(blob[20:20 + length])
    binary_offset = 20 + length
    binary_length, kind = struct.unpack_from('<II', blob, binary_offset)
    if kind != 0x004E4942 or binary_offset + 8 + binary_length != len(blob):
        raise ValueError('GLB must have one embedded BIN chunk')
    if len(tree.get('buffers', [])) != 1 or 'uri' in tree['buffers'][0]:
        raise ValueError('external/multiple glTF buffers are unsupported')
    return tree, blob[binary_offset + 8:]


def accessor_array(tree, binary, index):
    accessor = tree['accessors'][index]
    if 'sparse' in accessor or accessor.get('type') not in _WIDTHS:
        raise ValueError('unsupported glTF accessor')
    view = tree['bufferViews'][accessor['bufferView']]
    if view.get('buffer', 0) != 0:
        raise ValueError('external glTF accessor buffer')
    dtype = np.dtype(_DTYPES[accessor['componentType']])
    width = _WIDTHS[accessor['type']]
    stride = view.get('byteStride', width * dtype.itemsize)
    if stride < width * dtype.itemsize:
        raise ValueError('invalid glTF accessor stride')
    offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    return np.ndarray((accessor['count'], width), dtype=dtype, buffer=binary,
                      offset=offset, strides=(stride, dtype.itemsize)).copy()


def pack_glb(blob):
    """Combine only same-material/schema primitives with identity transforms.

    Positions, normals, UVs, colors and all other ordinary vertex attributes
    are copied byte-for-byte. Indices only receive the concatenated vertex
    offset. Materials, texture metadata and embedded encoded images survive
    unchanged; no atlas, remeshing, deduplication, or normal recomputation.
    """
    tree, binary = read_glb(blob)
    if any(tree.get(key) for key in ('animations', 'skins', 'extensionsUsed', 'extensionsRequired')):
        raise ValueError('animated/skinned/extended GLB is not a static laptop visual')
    roots = tree['scenes'][tree.get('scene', 0)].get('nodes', [])
    pending, visited, instances = list(roots), set(), []
    while pending:
        index = pending.pop(0)
        if index in visited:
            raise ValueError('shared/cyclic node hierarchy is unsupported')
        visited.add(index)
        node = tree['nodes'][index]
        if (not np.array_equal(node.get('matrix', np.eye(4).ravel()), np.eye(4).ravel())
                or not np.array_equal(node.get('translation', [0, 0, 0]), [0, 0, 0])
                or not np.array_equal(node.get('rotation', [0, 0, 0, 1]), [0, 0, 0, 1])
                or not np.array_equal(node.get('scale', [1, 1, 1]), [1, 1, 1])
                or any(key in node for key in ('skin', 'weights', 'camera', 'extensions'))):
            raise ValueError('laptop mesh packing requires identity static nodes')
        pending.extend(node.get('children', []))
        if 'mesh' in node:
            instances.append(tree['meshes'][node['mesh']])

    output = {key: deepcopy(tree[key]) for key in
              ('asset', 'materials', 'textures', 'samplers', 'images', 'extras') if key in tree}
    output.update(accessors=[], bufferViews=[], meshes=[], nodes=[], scenes=[{'nodes': []}], scene=0)
    packed = bytearray()

    def append_view(raw):
        packed.extend(b'\0' * (-len(packed) % 4))
        index = len(output['bufferViews'])
        output['bufferViews'].append({'buffer': 0, 'byteOffset': len(packed), 'byteLength': len(raw)})
        packed.extend(raw)
        return index

    for image in output.get('images', []):
        if 'uri' in image or 'bufferView' not in image:
            raise ValueError('laptop textures must be embedded')
        view = tree['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        image['bufferView'] = append_view(binary[start:start + view['byteLength']])

    groups = {}
    for mesh in instances:
        if mesh.get('weights'):
            raise ValueError('morph targets are unsupported')
        for primitive in mesh['primitives']:
            if (primitive.get('mode', 4) != 4
                    or any(key in primitive for key in ('targets', 'extensions', 'extras'))):
                raise ValueError('only ordinary triangle primitives can be packed')
            attributes = primitive['attributes']
            if 'POSITION' not in attributes:
                raise ValueError('primitive has no positions')
            signature = tuple((name, tree['accessors'][index]['componentType'],
                               tree['accessors'][index]['type'],
                               tree['accessors'][index].get('normalized', False))
                              for name, index in sorted(attributes.items()))
            groups.setdefault((primitive.get('material'), signature), []).append(primitive)

    def append_accessor(array, component, kind, normalized=False):
        index = len(output['accessors'])
        accessor = {'bufferView': append_view(array.tobytes()), 'componentType': component,
                    'count': len(array), 'type': kind}
        if normalized:
            accessor['normalized'] = True
        if array.size:
            accessor.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        output['accessors'].append(accessor)
        return index

    for group_index, ((material, signature), primitives) in enumerate(groups.items()):
        arrays = {name: [] for name, *_ in signature}
        indices, vertex_count = [], 0
        for primitive in primitives:
            attributes = {name: accessor_array(tree, binary, index)
                          for name, index in primitive['attributes'].items()}
            count = len(attributes['POSITION'])
            if any(len(array) != count for array in attributes.values()):
                raise ValueError('mismatched vertex attribute counts')
            local = (accessor_array(tree, binary, primitive['indices']).ravel()
                     if 'indices' in primitive else np.arange(count, dtype=np.uint32))
            if local.dtype.kind != 'u' or len(local) % 3 or np.any(local >= count):
                raise ValueError('invalid triangle indices')
            if vertex_count + count > np.iinfo(np.uint32).max:
                raise ValueError('packed mesh exceeds uint32 indexing')
            indices.append(local.astype('<u4') + vertex_count)
            vertex_count += count
            for name, array in attributes.items():
                arrays[name].append(array)
        combined = {'mode': 4, 'attributes': {}, 'indices': append_accessor(
            np.concatenate(indices).reshape(-1, 1).astype('<u4'), 5125, 'SCALAR')}
        if material is not None:
            combined['material'] = material
        for name, component, kind, normalized in signature:
            combined['attributes'][name] = append_accessor(
                np.concatenate(arrays[name]), component, kind, normalized)
        output['meshes'].append({'primitives': [combined]})
        output['nodes'].append({'mesh': group_index, 'name': f'packed_material_{group_index}'})
        output['scenes'][0]['nodes'].append(group_index)

    output['buffers'] = [{'byteLength': len(packed)}]
    packed.extend(b'\0' * (-len(packed) % 4))
    metadata = json.dumps(output, separators=(',', ':')).encode()
    metadata += b' ' * (-len(metadata) % 4)
    length = 12 + 8 + len(metadata) + 8 + len(packed)
    return (struct.pack('<IIIII', 0x46546C67, 2, length, len(metadata), 0x4E4F534A)
            + metadata + struct.pack('<II', len(packed), 0x004E4942) + packed)


def _atomic_write(destination, content):
    """Publish a complete cache file with a unique temporary name across nodes."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix=destination.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pack_urdf_visual_meshes(urdf_path):
    """Create new content-addressed visual caches; retain all physics XML."""
    source = Path(urdf_path)
    raw = source.read_bytes()
    tree = ET.fromstring(raw)
    visuals = []
    digest = hashlib.sha256(b'nyx-laptop-mesh-pack-v1\0' + raw)
    for visual in tree.findall('./link/visual/geometry/mesh'):
        path = source.parent / visual.attrib['filename']
        blob = path.read_bytes()
        digest.update(blob)
        visuals.append((visual, path, blob))
    if not visuals:
        raise ValueError('laptop URDF has no mesh visuals')
    key = digest.hexdigest()[:20]
    destination = source.parent / f'_nyx_laptop_pack_v1_{key}.urdf'
    for index, (visual, _path, blob) in enumerate(visuals):
        mesh_path = source.parent / f'_nyx_laptop_pack_v1_{key}_{index}.glb'
        if not mesh_path.exists():
            _atomic_write(mesh_path, pack_glb(blob))
        visual.set('filename', mesh_path.name)
    if not destination.exists():
        _atomic_write(destination, ET.tostring(tree, encoding='utf-8', xml_declaration=True))
    return destination
