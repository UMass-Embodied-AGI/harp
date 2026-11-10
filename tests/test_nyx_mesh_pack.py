"""Exact visual/physics preservation for the laptop-only NYX packer."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import struct
from threading import Barrier, Lock
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from envs.nyx_mesh_pack import accessor_array, pack_glb, pack_urdf_visual_meshes, read_glb
from envs import nyx_mesh_pack
from envs.genesis_compat import ensure_nyx_urdf
from envs.task_bases import neutral_avatar_clips


def encode(tree, binary):
    tree['buffers'] = [{'byteLength': len(binary)}]
    raw = json.dumps(tree).encode()
    raw += b' ' * (-len(raw) % 4)
    binary += b'\0' * (-len(binary) % 4)
    return (struct.pack('<IIIII', 0x46546C67, 2, 28 + len(raw) + len(binary), len(raw), 0x4E4F534A)
            + raw + struct.pack('<II', len(binary), 0x004E4942) + binary)


def example():
    tree = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
            'nodes': [{'children': [1, 2, 3]}], 'meshes': [], 'accessors': [], 'bufferViews': [],
            'materials': [{'name': 'paint'}, {'name': 'texture', 'doubleSided': True,
                           'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}],
            'textures': [{'source': 0}], 'images': [{'bufferView': 0, 'mimeType': 'image/png'}]}
    binary = bytearray(b'exact encoded pixels')
    tree['bufferViews'].append({'buffer': 0, 'byteOffset': 0, 'byteLength': len(binary)})
    for i in range(3):
        attrs = {}
        for name, kind, values in (
            ('POSITION', 'VEC3', [[i, 0, 0], [i, 1, 0], [i, 0, 1]]),
            ('NORMAL', 'VEC3', [[1, 0, 0]] * 3),
            ('TEXCOORD_0', 'VEC2', [[.13, .14], [.15, .16], [.17, .18]]),
        ):
            binary.extend(b'\0' * (-len(binary) % 4))
            data = np.asarray(values, dtype='<f4').tobytes()
            view = len(tree['bufferViews'])
            tree['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)})
            binary.extend(data)
            attrs[name] = len(tree['accessors'])
            tree['accessors'].append({'bufferView': view, 'componentType': 5126, 'type': kind, 'count': 3})
        tree['meshes'].append({'primitives': [{'attributes': attrs, 'material': i % 2}]})
        tree['nodes'].append({'mesh': i})
    return encode(tree, bytes(binary))


def visual_fingerprint(blob):
    tree, binary = read_glb(blob)
    triangles = Counter()
    for mesh in tree['meshes']:
        for primitive in mesh['primitives']:
            values = {name: accessor_array(tree, binary, idx)
                      for name, idx in primitive['attributes'].items()}
            count = len(values['POSITION'])
            indices = (accessor_array(tree, binary, primitive['indices']).ravel()
                       if 'indices' in primitive else np.arange(count))
            for face in indices.reshape(-1, 3):
                signature = tuple((name, array.dtype.str, array[face].tobytes())
                                  for name, array in sorted(values.items()))
                triangles[(primitive.get('material'), signature)] += 1
    images = []
    for image in tree.get('images', []):
        image = dict(image)
        view = tree['bufferViews'][image.pop('bufferView')]
        offset = view.get('byteOffset', 0)
        images.append((image, binary[offset:offset + view['byteLength']]))
    return triangles, tree.get('materials'), tree.get('textures'), tree.get('samplers'), images


def test_material_uv_normals_triangles_and_encoded_images_unchanged():
    blob = example()
    packed = pack_glb(blob)
    assert visual_fingerprint(blob) == visual_fingerprint(packed)
    before, _ = read_glb(blob)
    after, _ = read_glb(packed)
    assert len(before['meshes']) == 3 and len(after['meshes']) == 2
    assert len(after['nodes']) == 2


@pytest.mark.parametrize('change', ['translate', 'rotate', 'scale', 'matrix', 'skin', 'mode', 'animation'])
def test_unhandled_features_fail_without_lossy_conversion(change):
    tree, binary = read_glb(example())
    if change == 'translate':
        tree['nodes'][1]['translation'] = [1, 0, 0]
    elif change == 'rotate':
        tree['nodes'][1]['rotation'] = [0, 0, 1, 0]
    elif change == 'scale':
        tree['nodes'][1]['scale'] = [2, 2, 2]
    elif change == 'matrix':
        matrix = np.eye(4).ravel().tolist()
        matrix[12] = 1
        tree['nodes'][1]['matrix'] = matrix
    elif change == 'skin':
        tree['nodes'][1]['skin'] = 0
    elif change == 'mode':
        tree['meshes'][0]['primitives'][0]['mode'] = 5
    else:
        tree['animations'] = [{'channels': []}]
    with pytest.raises(ValueError):
        pack_glb(encode(tree, binary))


def test_urdf_cache_changes_only_visual_filenames_and_keeps_old_files(tmp_path):
    mesh = tmp_path / 'visual.glb'
    mesh.write_bytes(example())
    source = tmp_path / 'original.urdf'
    source.write_text('<robot name="laptop"><link name="base">'
                      '<inertial><mass value="2"/></inertial>'
                      '<collision><geometry><mesh filename="collision.obj" scale="0.28 0.28 0.28"/></geometry></collision>'
                      '<visual><origin xyz="1 2 3" rpy="0 1 0"/><geometry>'
                      '<mesh filename="visual.glb" scale="0.28 0.28 0.28"/></geometry></visual>'
                      '</link><joint name="lid" type="revolute"><limit lower="0" upper="1.2"/></joint></robot>')
    original = source.read_bytes()
    destination = pack_urdf_visual_meshes(source)
    assert destination != source and source.read_bytes() == original
    assert mesh.read_bytes() == example()
    tree = ET.parse(destination)
    packed_mesh = tree.find('./link/visual/geometry/mesh')
    assert visual_fingerprint((tmp_path / packed_mesh.get('filename')).read_bytes()) == visual_fingerprint(example())
    packed_mesh.set('filename', 'visual.glb')
    assert ET.tostring(tree.getroot()) == ET.tostring(ET.fromstring(original))
    assert pack_urdf_visual_meshes(source) == destination
    tree, binary = read_glb(example())
    tree['materials'][0]['name'] = 'changed'
    mesh.write_bytes(encode(tree, binary))
    assert pack_urdf_visual_meshes(source) != destination
    assert destination.exists()


def test_atomic_cache_writers_have_unique_temporary_paths(tmp_path, monkeypatch):
    destination = tmp_path / 'shared.glb'
    content = pack_glb(example())
    barrier, lock, names = Barrier(4), Lock(), []
    replace = nyx_mesh_pack.os.replace

    def simultaneous_replace(source, target):
        with lock:
            names.append(source)
        barrier.wait(timeout=10)
        replace(source, target)

    monkeypatch.setattr(nyx_mesh_pack.os, 'replace', simultaneous_replace)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: nyx_mesh_pack._atomic_write(destination, content), range(4)))
    assert len(set(names)) == 4
    assert destination.read_bytes() == content
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('flag', [None, False, 1, 'true', True])
@pytest.mark.parametrize('renderer', ['nyx', 'rasterizer'])
@pytest.mark.parametrize('asset', ['015_laptop', 'sapien-chair-179'])
def test_dispatch_on_unless_false_nyx_laptop(flag, renderer, asset, monkeypatch):
    class Task(neutral_avatar_clips._NeutralAvatarClipsMixin):
        pass

    task = Task()
    task.config = {'renderer': renderer}
    if flag is not None:
        task.config['nyx_laptop_mesh_pack_v1'] = flag
    task.scene = object()
    pose, calls = object(), []
    source = Path('/assets/objects') / asset / 'mobility.urdf'
    converted = source.with_name('mobility_nyx.urdf')
    packed = source.with_name('packed.urdf')
    monkeypatch.setattr(neutral_avatar_clips, 'ensure_nyx_urdf', lambda path, scale: converted)
    monkeypatch.setattr(nyx_mesh_pack, 'pack_urdf_visual_meshes',
                        lambda path: calls.append(path) or packed)
    monkeypatch.setattr(neutral_avatar_clips, 'load_urdf', lambda *args, **kwargs: (args, kwargs))
    args, kwargs = task._neutral_load_prop_urdf(source, pose, .28)
    should_pack = flag is not False and renderer == 'nyx' and asset == '015_laptop'  # default on
    assert calls == ([converted] if should_pack else [])
    assert args == (task.scene, packed if should_pack else converted if renderer == 'nyx' else source, pose)
    assert kwargs == {'scale': 1.0 if renderer == 'nyx' else .28,
                      'fix_root': True, 'merge_fixed_links': False if renderer == 'nyx' else None}


@pytest.fixture(scope='module')
def authored_laptop_visuals(tmp_path_factory):
    """Build fixtures from shipped inputs, never an existing NYX cache.

    Conversion writes only into this temporary copy. Explicit dependency checks
    prevent trimesh's missing-material/texture fallbacks from making a lossy
    source appear to pass the packer's preservation assertions.
    """
    root = Path(__file__).resolve().parents[1]
    source = root / 'assets/objects/015_laptop/9960'
    staged = tmp_path_factory.mktemp('authored_laptop')
    shutil.copy2(source / 'mobility.urdf', staged / 'mobility.urdf')
    for directory in ('textured_objs', 'images'):
        shutil.copytree(source / directory, staged / directory)
    assert not list(staged.glob('_nyx*'))

    for mesh in ET.parse(staged / 'mobility.urdf').iter('mesh'):
        path = staged / mesh.attrib['filename']
        assert path.is_file(), f'missing authored laptop mesh: {path}'
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == 'mtllib':
                for name in fields[1:]:
                    material = path.parent / name
                    assert material.is_file(), f'missing authored laptop material: {material}'
                    for material_line in material.read_text().splitlines():
                        texture = material_line.split()
                        if texture and texture[0].startswith('map_'):
                            assert len(texture) == 2, f'unsupported texture reference: {material_line}'
                            image = material.parent / texture[1]
                            assert image.is_file(), f'missing authored laptop texture: {image}'

    converted = ensure_nyx_urdf(staged / 'mobility.urdf', scale=.28)
    assert converted.parent == staged and converted != staged / 'mobility.urdf'
    visuals = {}
    for link in ET.parse(converted).findall('link'):
        meshes = link.findall('./visual/geometry/mesh')
        assert len(meshes) == 1
        path = converted.parent / meshes[0].attrib['filename']
        assert path.parent == staged and path.suffix == '.glb'
        visuals[link.attrib['name']] = path.read_bytes()
    assert set(visuals) == {'base', 'link_0'}
    return visuals


@pytest.mark.parametrize('name,expected_max_meshes', [('base', 4), ('link_0', 4)])
def test_actual_laptop_preserves_every_triangle_attribute_and_image(
    name, expected_max_meshes, authored_laptop_visuals,
):
    blob = authored_laptop_visuals[name]
    packed = pack_glb(blob)
    assert visual_fingerprint(blob) == visual_fingerprint(packed)
    tree, _ = read_glb(packed)
    assert len(tree['meshes']) <= expected_max_meshes
