import sys,importlib,inspect,pkgutil,collections
import apeGmsh.opensees as O
from apeGmsh.opensees._internal import types as T
for m in pkgutil.walk_packages(O.__path__, 'apeGmsh.opensees.'):
    try: importlib.import_module(m.name)
    except Exception as e: pass
def subs(base):
    out=set(); st=[base]
    while st:
        b=st.pop()
        for s in b.__subclasses__():
            st.append(s)
            if not s.__name__.startswith('_') and not inspect.isabstract(s) and s.__module__.startswith('apeGmsh.opensees'): out.add(s)
    return out
sys.path.insert(0,'.')
from tests.opensees.contract.test_primitive_base import ALL_PRIMITIVES
allp=set(ALL_PRIMITIVES)
prims=subs(T.Primitive)
print('concrete Primitive subclasses:',len(prims),' in ALL_PRIMITIVES:',len(prims&allp),' missing:',len(prims-allp), ' ALL_PRIMITIVES len', len(ALL_PRIMITIVES))
nd=[p for p in prims if p.__module__.endswith('material.nd')]
print('nd:',len(nd),'missing',len([p for p in nd if p not in allp]))
key=lambda c:(c.__module__,c.__qualname__)
P={key(p) for p in prims}; A={key(p) for p in allp}
print('dedup: concrete',len(P),'listed',len(P&A),'missing',len(P-A))
c=collections.Counter(m for m,_ in P-A)
tot=collections.Counter(m for m,_ in P)
for k,v in sorted(c.items(), key=lambda kv:-kv[1]): print(f'   {v:3d}/{tot[k]:3d} {k}')
