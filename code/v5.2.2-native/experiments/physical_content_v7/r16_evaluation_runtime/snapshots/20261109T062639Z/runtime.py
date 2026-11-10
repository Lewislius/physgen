"""Preserve trained source/weights; supply evaluation GPU policy and CFG API fix."""
from pathlib import Path
import functools,importlib.util,sys

HERE=Path(__file__).resolve().parent


def activate(source,pool):
    source=Path(source).resolve();sys.path.insert(0,str(source))
    import bootstrap
    import physical_scaling.common as scaling_common
    import physical_content.common as content_common
    from gpu_runtime import gpu_guard
    guard=functools.partial(gpu_guard,pool=pool)
    scaling_common.gpu_guard=guard;content_common.gpu_guard=guard
    spec=importlib.util.spec_from_file_location('guidance',HERE/'guidance_fix.py')
    module=importlib.util.module_from_spec(spec);sys.modules['guidance']=module;spec.loader.exec_module(module)
    return guard


def stage_command(command,source,pool):
    command=list(map(str,command))
    if command[0]!=sys.executable:return command
    index=next(i for i,value in enumerate(command[1:],1) if value.endswith('.py'))
    script=Path(command[index])
    if script.name in {'train.py','train_only.py','gpu_preflight.py'}:
        raise ValueError('Evaluation recovery must never start training')
    return command[:index]+[str(HERE/'execute_stage.py'),str(source),pool,str(script)]+command[index+1:]
