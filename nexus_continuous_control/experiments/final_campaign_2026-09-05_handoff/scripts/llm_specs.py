"""Strict bounded specification generation and a validated adapter to the supplied DSL.

No generated Python is executed. Invalid samples are retained as failures. The
small existing model is retained; this is not a comparison of model families.
"""
from __future__ import annotations
import argparse, ast, json, math
from pathlib import Path
from common import write_json, atomic

FIELDS={'cheetah':('forward_velocity','torso_pitch','joint_speed'),
        'walker':('height','torso_pitch','forward_velocity','joint_speed')}
COUNTS={'cheetah':3,'walker':4}
TYPES={'negative_distance','positive_velocity','target_height','binary_bonus','action_penalty','posture_penalty'}
ALLOWED=(ast.Expression,ast.BoolOp,ast.And,ast.Or,ast.UnaryOp,ast.Not,ast.USub,ast.UAdd,
         ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Compare,ast.Gt,ast.GtE,ast.Lt,ast.LtE,
         ast.Eq,ast.NotEq,ast.Call,ast.Name,ast.Load,ast.Constant)

def validate(spec:dict,task:str)->None:
    if task not in FIELDS: raise ValueError('Only cheetah and walker are in the frozen LLM study')
    skills=spec.get('skills')
    if not isinstance(skills,list) or len(skills)!=COUNTS[task]: raise ValueError(f'Exactly {COUNTS[task]} skills required')
    if len({s.get('name') for s in skills})!=len(skills): raise ValueError('Unique names required')
    always=False
    for s in skills:
        if not isinstance(s.get('name'),str) or not s['name'].strip(): raise ValueError('Missing skill name')
        rule=s.get('activation_rule')
        if not isinstance(rule,str) or not rule: raise ValueError('Missing activation_rule')
        tree=ast.parse(rule,mode='eval'); always|=isinstance(tree.body,ast.Constant) and tree.body.value is True
        for n in ast.walk(tree):
            if not isinstance(n,ALLOWED): raise ValueError(f'Unsupported rule syntax: {type(n).__name__}')
            if isinstance(n,ast.Name) and n.id not in (*FIELDS[task],'abs','min','max'):
                raise ValueError(f'Unknown rule field {n.id}')
            if isinstance(n,ast.Constant) and (not isinstance(n.value,(int,float,bool)) or not math.isfinite(float(n.value))):
                raise ValueError('Only finite numeric/bool rule constants')
            if isinstance(n,ast.Call):
                if not isinstance(n.func,ast.Name) or n.func.id not in ('abs','min','max') or n.keywords:
                    raise ValueError('Only abs/min/max calls')
                if len(n.args)!=(1 if n.func.id=='abs' else 2): raise ValueError('Wrong call arity')
        terms=s.get('reward_terms')
        if not isinstance(terms,list) or not 1<=len(terms)<=6: raise ValueError('One to six reward terms per skill')
        for t in terms:
            if set(t)-{'type','weight','lhs','rhs','threshold'}: raise ValueError('Unknown reward-term keys')
            if t.get('type') not in TYPES: raise ValueError(f'Unsupported reward type {t.get("type")}')
            w=t.get('weight',1.)
            if not isinstance(w,(int,float)) or isinstance(w,bool) or not 0<float(w)<=10:
                raise ValueError('Weights must be positive and <=10; zero is rejected, never changed into one')
            if t.get('type')!='action_penalty' and t.get('lhs') not in FIELDS[task]: raise ValueError('Known lhs required')
            if 'rhs' in t and t['rhs'] not in FIELDS[task]: raise ValueError('rhs must be a known field; constants use threshold')
            if 'threshold' in t and (not isinstance(t['threshold'],(float,int)) or not math.isfinite(t['threshold'])):
                raise ValueError('Finite numeric threshold required')
            if 'rhs' in t and 'threshold' in t: raise ValueError('threshold would be ignored when rhs is present')
            typ=t['type']
            if 'rhs' in t and typ!='negative_distance': raise ValueError('rhs is consumed only by negative_distance')
            if 'threshold' in t and typ not in ('negative_distance','target_height','binary_bonus'):
                raise ValueError('threshold would be ignored for this type')
            if typ in ('target_height','binary_bonus') and 'threshold' not in t: raise ValueError('Explicit threshold required')
            if typ=='target_height' and t['threshold']<=0: raise ValueError('target_height threshold must be positive')
            if typ=='action_penalty' and set(t)-{'type','weight'}: raise ValueError('action_penalty has no field arguments')
    if not always: raise ValueError('One skill must explicitly use activation_rule="True" to cover all states')

def install_policy(cfg,consumer):
    if 'CAMPAIGN_SPEC_PAYLOAD' not in cfg: return
    from nexus_continuous.policies.registry import load_policy_module
    from nexus_continuous.llm.interpreter import make_policy_module
    env=cfg['ENV_NAME']; task={'CheetahRun':'cheetah','WalkerWalk':'walker'}[env]
    spec=cfg['CAMPAIGN_SPEC_PAYLOAD']; validate(spec,task)
    hand=load_policy_module(cfg['TASK_POLICY'])
    def fields(obs,info=None):
        f=hand._features(obs,info)
        if task=='cheetah':
            vx,pitch,speed=f
            return dict(forward_velocity=vx,torso_pitch=pitch,joint_speed=speed)
        height,pitch,vx,speed=f
        return dict(height=height,torso_pitch=pitch,forward_velocity=vx,joint_speed=speed)
    module=make_policy_module(spec,FIELDS[task],task_metrics_fn=hand.task_metrics,
                              field_fn=fields,mask_mode='strict')
    original=consumer.load_policy_module
    consumer.load_policy_module=lambda name:module if name=='llm_generated' else original(name)

PROMPT='''Design a continuous-control NEXUS skillset. Return JSON only: {"skills": [
 {"name":"...", "activation_rule":"...", "reward_terms":[{"type":"...","weight":1.0,"lhs":"..."}]}]}.
Every scalar field is the same semantic feature supplied to the hand-designed controller.
Fields and units: forward_velocity in m/s, torso_pitch in radians, joint_speed mean absolute joint
angular speed in rad/s, and height in metres only when listed. No hidden fields or environmental
reward are available. Use EXACTLY the requested skill count and no extra term keys.
One skill must have activation_rule "True". Others may overlap. NeSy chooses max learned meta-Q
among allowed skills every step. Define simple complementary goals, not named copies.
Rules: numeric comparisons, and/or/not, + - *, abs(x), min(x,y), max(x,y). No division.
Rewards sum terms. All weights >0 and <=10; penalty types negate their weight internally.
Allowed vocabulary and EXACT executable semantics:
negative_distance: -weight*abs(lhs-rhs), or -weight*abs(lhs-threshold), or -weight*abs(lhs).
positive_velocity: weight*lhs.
target_height: weight*clip(lhs/threshold,0,1), requires positive threshold (not rhs).
binary_bonus: weight*(lhs>threshold), requires lhs and threshold.
action_penalty: -weight*sum(action**2), has neither lhs nor rhs nor threshold.
posture_penalty: -weight*abs(lhs).
rhs must name another available field, never a number. Constants use threshold when supported.
A terminal penalty of 1 is subtracted from every skill. Produce between one and six terms per skill.
'''

def main():
    from common import ROOT
    if (ROOT/'INSTALLING').exists():
        raise RuntimeError('Installation incomplete; no model download or generation started')
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    lock=sub.add_parser('lock-model'); lock.add_argument('--out',required=True)
    gen=sub.add_parser('generate'); gen.add_argument('--model-lock',required=True)
    gen.add_argument('--task',choices=FIELDS,required=True); gen.add_argument('--family',type=int,choices=range(3),required=True)
    gen.add_argument('--condition',choices=('initial','refined','resample'),required=True)
    gen.add_argument('--initial'); gen.add_argument('--feedback'); gen.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='lock-model':
        from huggingface_hub import HfApi,snapshot_download
        model='Qwen/Qwen2.5-1.5B-Instruct'; rev=HfApi().model_info(model).sha
        local=snapshot_download(model,revision=rev)
        write_json(Path(a.out),dict(model=model,revision=rev,local_path=local,
            temperature=.7,top_p=.9,max_new_tokens=4096,max_syntax_repairs=2)); return
    lock=json.loads(Path(a.model_lock).read_text()); out=Path(a.out)
    if out.exists() or out.with_suffix('.generation.json').exists(): raise FileExistsError(out)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    tok=AutoTokenizer.from_pretrained(lock['local_path'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(lock['local_path'],local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto').eval()
    seed={'initial':2000,'refined':4000,'resample':3000}[a.condition]+a.family
    task_text='Run forward quickly while maintaining posture.' if a.task=='cheetah' else 'Walk forward around 1 m/s while remaining upright near 1.2 m torso height.'
    messages=[dict(role='system',content=PROMPT),dict(role='user',content=
        f'Task: {a.task}. {task_text} Available fields: {FIELDS[a.task]}. Skill count: {COUNTS[a.task]}.')]
    if a.condition=='refined':
        if not a.initial or not a.feedback: raise ValueError('Refinement needs initial JSON and held-out PILOT validation summary')
        messages += [dict(role='assistant',content=Path(a.initial).read_text()),
            dict(role='user',content='Revise this SAME proposal once using these pilot validation metrics. '
                 'Do not add skills. All fields and reward constraints stay the same.\n'+Path(a.feedback).read_text())]
    attempts=[]
    for attempt in range(3):
        set_seed(seed+10000*attempt)
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tok(text,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            result=model.generate(**inputs,do_sample=True,temperature=.7,top_p=.9,
                                  max_new_tokens=4096,pad_token_id=tok.eos_token_id)
        raw=tok.decode(result[0,inputs.input_ids.shape[-1]:],skip_special_tokens=True)
        try:
            cleaned=raw.strip()
            if cleaned.startswith('```'):
                cleaned='\n'.join(cleaned.splitlines()[1:-1])
            spec=json.loads(cleaned); validate(spec,a.task); error=None
        except Exception as e: error=f'{type(e).__name__}: {e}'
        attempts.append(dict(seed=seed+10000*attempt,prompt=messages.copy(),raw=raw,error=error))
        if error is None:
            write_json(out,spec); break
        messages += [dict(role='assistant',content=raw),dict(role='user',content=
            'Syntax/type validation failed: '+error+'. Repair only schema/type errors; keep your proposed goals.')]
    write_json(out.with_suffix('.generation.json'),dict(model=lock,task=a.task,family=a.family,
        condition=a.condition,attempts=attempts,valid=error is None,
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__))
    if error is not None: raise SystemExit('Invalid proposal after two bounded repairs: recorded failure; no replacement sample')
if __name__=='__main__': main()
