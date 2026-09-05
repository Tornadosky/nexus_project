"""CPU tests: matrix arithmetic, schema rejection, non-destructive IO, source anchors.
These do NOT claim that MuJoCo/JAX GPU training has been executed.
"""
import ast,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import atomic,replace_once
from llm_specs import validate
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_code_parses(self):
        for path in (ROOT/'scripts').glob('*.py'): ast.parse(path.read_text(),filename=str(path))
    def test_matrix(self):
        rows=json.loads((ROOT/'plan/matrix.json').read_text()); self.assertEqual(len(rows),142)
        self.assertEqual(len({r['id'] for r in rows}),142)
        self.assertEqual(sum(r['budget'] for r in rows),7073792000)
        for r in rows:
            if r['engine']=='state': self.assertEqual(r['budget']%131072,0)
            if r['engine']=='ppo':
                unit=983040 if r['task']=='hopper' else 163840
                self.assertEqual(r['budget']%(10*unit),0)
    def test_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'keep'; atomic(path,b'original')
            with self.assertRaises(FileExistsError): atomic(path,b'changed')
            self.assertEqual(path.read_bytes(),b'original')
    def test_anchor(self):
        self.assertEqual(replace_once('abc','b','x'),'axc')
        with self.assertRaises(RuntimeError): replace_once('bb','b','x')
    def test_validator(self):
        def spec(): return {'skills':[{'name':f's{i}','activation_rule':'True' if i==0 else 'abs(torso_pitch) < 0.4',
            'reward_terms':[{'type':'positive_velocity','lhs':'forward_velocity','weight':1.}]} for i in range(3)]}
        validate(spec(),'cheetah')
        x=spec(); x['skills'][1]['reward_terms'][0]['type']='invented'
        with self.assertRaises(ValueError): validate(x,'cheetah')
        x=spec(); x['skills'][0]['reward_terms'][0]['weight']=0
        with self.assertRaises(ValueError): validate(x,'cheetah')
        x=spec(); x['skills'][1]['activation_rule']='__import__("os").system("false")'
        with self.assertRaises(ValueError): validate(x,'cheetah')
        x=spec(); x['skills'][0]['reward_terms'][0]['lhs']='unknown'
        with self.assertRaises(ValueError): validate(x,'cheetah')
if __name__=='__main__': unittest.main()
