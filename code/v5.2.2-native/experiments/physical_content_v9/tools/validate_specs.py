"""Validate plan/config contracts only. Does not instantiate or train Wan."""
from pathlib import Path
import json, sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name, cond):
    checks.append({'name':name,'passed':bool(cond)})
    if not cond: raise AssertionError(name)

def main():
    expected_targets={'self_attn.q','self_attn.k','self_attn.v','self_attn.o','ffn.0','ffn.2'}
    for k in 'ABCD':
        path=ROOT/'configs'/f'{k.lower()}_fresh1000_batch4.yaml'
        c=yaml.safe_load(path.read_text(encoding='utf8'))
        check(k+':variant',c['variant']==k)
        check(k+':official_fresh',c['initialization']['mode']=='official_fresh' and c['initialization']['project_checkpoint'] is None and not c['initialization']['load_project_lora'])
        check(k+':1000_updates',c['train']['max_updates']==1000 and c['train']['freeze_wan_lora_updates']==0)
        check(k+':effective_batch4',c['train']['effective_global_batch']==4==c['train']['world_size']*c['train']['microbatch_per_device']*c['train']['gradient_accumulation'])
        check(k+':base_and_text_frozen',not c['model']['native_base_trainable'] and not c['model']['native_text_lora'] and not c['model']['native_text_params_trainable'])
        l=c['model']['native_video_lora']
        check(k+':video_lora_allowlist',set(l['targets'])==expected_targets and l['layers']==list(range(30)) and l['active_updates']==[1,1000])
        check(k+':no_post_or_extra_views',not c['train']['post_training'] and c['train']['extra_training_views']==0)
        check(k+':no_optional_losses',all(c['loss'][x]==0 for x in ('preference_weight','reward_weight','temporal_weight','rgb_weight','teacher_feature_weight')))
        check(k+':generation_loss_weights',abs(c['loss']['fm_weight']+c['loss']['tail_weight']-1)<1e-12 and c['loss']['tail_fraction']==.25)
        check(k+':route_raw_and_decay',c['loss']['route']['target']=='raw_route_only' and c['loss']['route']['zero_at_update']==320)
        every=c['model']['site_period']
        check(k+':sites',c['model']['sites_zero_based']==[i for i in range(30) if (i+1)%every==0])
        check(k+':zero_final_writer',c['model']['write_head']['init']=='zeros' and c['model']['write_head']['inference_multiplier']==1.0)
        check(k+':full_width',c['model']['width']==3072 and c['model']['q_tokens']==256 and c['model']['ffn_width']==14336)
        check(k+':sampling',c['inference']['steps']==50 and c['inference']['cfg']==5.0 and c['inference']['best_of_n']==1)
        plan=(ROOT/f'PLAN_{k}_FRESH1000.md').read_text(encoding='utf8')
        marker='# 本方案默认YAML\n\n```yaml\n'
        embedded=plan.split(marker)[1].split('```',1)[0]
        check(k+':embedded_config_matches',yaml.safe_load(embedded)==c)
    report={'kind':'specification_validation_only','checks':len(checks),'passed':sum(x['passed'] for x in checks),'details':checks,'wan_gpu_training_run':False}
    (ROOT/'SPEC_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('kind','checks','passed','wan_gpu_training_run')},ensure_ascii=False))

if __name__=='__main__':main()
